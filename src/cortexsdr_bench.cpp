#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct CommandResult {
    int exit_code = -1;
    double elapsed_ms = 0.0;
    std::string stdout_text;
    std::string stderr_text;
    std::string command;
};

static std::string quoteArg(const std::string& v) {
    std::string out = "\"";
    for (char c : v) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

static std::string readTextFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static CommandResult runCommandCapture(
    const std::vector<std::string>& args,
    const fs::path& stdout_file,
    const fs::path& stderr_file) {
    CommandResult result;
    if (args.empty()) {
        result.stderr_text = "Empty command";
        return result;
    }

    std::ostringstream cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            cmd << ' ';
        }
        cmd << quoteArg(args[i]);
    }
    cmd << " > " << quoteArg(stdout_file.string()) << " 2> " << quoteArg(stderr_file.string());
    result.command = cmd.str();

    const auto start = std::chrono::steady_clock::now();
    result.exit_code = std::system(result.command.c_str());
    const auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.stdout_text = readTextFile(stdout_file);
    result.stderr_text = readTextFile(stderr_file);
    return result;
}

static std::optional<double> extractDouble(const std::string& text, const std::regex& pattern) {
    std::smatch m;
    if (std::regex_search(text, m, pattern) && m.size() > 1) {
        try {
            return std::stod(m[1].str());
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

static std::optional<double> jsonNumberToDouble(const json& obj, const char* key) {
    if (!obj.contains(key) || !obj[key].is_number()) {
        return std::nullopt;
    }
    return obj[key].get<double>();
}

static double toMB(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

static std::optional<double> rateMBPerSec(double bytes, double elapsed_ms) {
    if (elapsed_ms <= 0.0) {
        return std::nullopt;
    }
    return toMB(bytes) / (elapsed_ms / 1000.0);
}

static size_t estimateTokensSimple(const std::string& text) {
    size_t count = 0;
    bool in_token = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) || std::ispunct(static_cast<unsigned char>(c))) {
            if (in_token) {
                ++count;
                in_token = false;
            }
            if (std::ispunct(static_cast<unsigned char>(c))) {
                ++count;
            }
        } else {
            in_token = true;
        }
    }
    if (in_token) {
        ++count;
    }
    return count;
}

static std::uintmax_t directorySizeBytes(const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return 0;
    }
    std::uintmax_t total = 0;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            total += entry.file_size();
        }
    }
    return total;
}

static void printUsage(const char* program) {
    std::cout << "CortexSDR Benchmark Runner\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program << " --config <path-to-suite.json> [--output <run.json>]\n\n";
    std::cout << "Config contains binary paths + tests for compress/extract/inference.\n";
}

static json summarizeSeries(const std::vector<double>& values) {
    if (values.empty()) {
        return {{"count", 0}};
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    const auto percentile = [&sorted, n](double p) -> double {
        const double idx = p * static_cast<double>(n - 1);
        const size_t lo = static_cast<size_t>(std::floor(idx));
        const size_t hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) {
            return sorted[lo];
        }
        const double alpha = idx - static_cast<double>(lo);
        return sorted[lo] * (1.0 - alpha) + sorted[hi] * alpha;
    };

    double sum = 0.0;
    for (double v : sorted) {
        sum += v;
    }
    return {
        {"count", n},
        {"min", sorted.front()},
        {"max", sorted.back()},
        {"mean", sum / static_cast<double>(n)},
        {"p50", percentile(0.50)},
        {"p95", percentile(0.95)},
    };
}

int main(int argc, char* argv[]) {
    std::string config_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    const fs::path cfg_path = fs::absolute(config_path);
    json cfg;
    try {
        cfg = json::parse(readTextFile(cfg_path));
    } catch (const std::exception& ex) {
        std::cerr << "Failed to parse config JSON: " << ex.what() << "\n";
        return 1;
    }

    const fs::path project_root = fs::absolute(cfg.value("project_root", "."));
    const std::string ai_cli = cfg["binaries"].value("ai_compression_cli", "build/Debug/cortexsdr_ai_compression_cli.exe");
    const std::string text_cli = cfg["binaries"].value("text_cli", "build/Debug/cortexsdr_text_cli.exe");
    const int default_repeats = cfg.value("repeats", 1);
    const double default_sparsity = cfg.value("default_sparsity", 0.02);
    const int default_warmup_runs = cfg.value("warmup_runs", 0);

    const fs::path ai_cli_path = fs::absolute(project_root / ai_cli);
    const fs::path text_cli_path = fs::absolute(project_root / text_cli);
    const fs::path temp_dir = fs::absolute(project_root / "benchmarks" / "tmp");
    fs::create_directories(temp_dir);

    const auto ts = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    fs::path out_path = output_path.empty()
                            ? fs::absolute(project_root / "benchmarks" / "results" / ("run_" + std::to_string(ts) + ".json"))
                            : fs::absolute(output_path);
    fs::create_directories(out_path.parent_path());

    json run;
    run["runner"] = "cortexsdr_bench";
    run["project_root"] = project_root.string();
    run["config"] = cfg_path.string();
    run["binaries"] = {{"ai_compression_cli", ai_cli_path.string()}, {"text_cli", text_cli_path.string()}};
    run["results"] = json::array();

    const std::regex original_re(R"(Original size:\s*([0-9]+)\s*bytes)", std::regex::icase);
    const std::regex compressed_re(R"(Compressed size:\s*([0-9]+)\s*bytes)", std::regex::icase);
    const std::regex ratio_re(R"(Compression ratio:\s*([0-9]+(?:\.[0-9]+)?)\s*:1)", std::regex::icase);
    const std::regex ctime_re(R"(Compression time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms)", std::regex::icase);

    for (const auto& test : cfg["tests"]) {
        const std::string name = test.value("name", "unnamed");
        const std::string format = test.value("format", "onnx");
        const fs::path model_path = fs::absolute(project_root / test.value("model_path", ""));
        const fs::path sdr_path = fs::absolute(project_root / test.value("sdr_path", "benchmarks/artifacts/model.sdr"));
        const fs::path extract_dir = fs::absolute(project_root / test.value("extract_dir", "benchmarks/artifacts/extract"));
        const int repeats = test.value("repeats", default_repeats);
        const int warmup_runs = test.value("warmup_runs", default_warmup_runs);
        const double sparsity = test.value("sparsity", default_sparsity);
        const bool run_compress = test.value("run_compress", true);
        const bool run_extract = test.value("run_extract", true);
        const bool run_inference = test.value("run_inference", false);
        const std::string prompt = test.value("prompt", "Hello from benchmark");
        const int max_length = test.value("max_length", 128);
        const std::string mode = test.value("inference_mode", "on-demand");

        for (int i = 0; i < repeats; ++i) {
            json row;
            row["name"] = name;
            row["repeat"] = i + 1;
            row["format"] = format;
            row["model_path"] = model_path.string();
            row["sdr_path"] = sdr_path.string();
            row["status"] = "ok";

            if (!fs::exists(model_path)) {
                row["status"] = "error";
                row["error"] = "Model path not found";
                run["results"].push_back(row);
                continue;
            }

            fs::create_directories(sdr_path.parent_path());
            fs::create_directories(extract_dir);

            // Warmup runs are executed but not recorded.
            for (int warm = 0; warm < warmup_runs; ++warm) {
                if (run_compress) {
                    const fs::path wcoutf = temp_dir / (name + "_warmup_compress_out.txt");
                    const fs::path wcerrf = temp_dir / (name + "_warmup_compress_err.txt");
                    (void)runCommandCapture(
                        {ai_cli_path.string(), "-c", model_path.string(), format, sdr_path.string(), std::to_string(sparsity)},
                        wcoutf,
                        wcerrf);
                }
                if (run_extract) {
                    const fs::path weoutf = temp_dir / (name + "_warmup_extract_out.txt");
                    const fs::path weerrf = temp_dir / (name + "_warmup_extract_err.txt");
                    if (fs::exists(extract_dir)) {
                        fs::remove_all(extract_dir);
                    }
                    fs::create_directories(extract_dir);
                    (void)runCommandCapture(
                        {ai_cli_path.string(), "-x", sdr_path.string(), extract_dir.string(), std::to_string(sparsity)},
                        weoutf,
                        weerrf);
                }
                if (run_inference) {
                    const fs::path wprof = temp_dir / (name + "_warmup_profile.json");
                    const fs::path wioutf = temp_dir / (name + "_warmup_inference_out.txt");
                    const fs::path wierrf = temp_dir / (name + "_warmup_inference_err.txt");
                    if (fs::exists(wprof)) {
                        fs::remove(wprof);
                    }
                    std::vector<std::string> wcmd = {text_cli_path.string()};
                    if (mode == "legacy") {
                        wcmd.push_back("--legacy");
                    } else {
                        wcmd.push_back("--on-demand");
                    }
                    wcmd.push_back("-p");
                    wcmd.push_back(prompt);
                    wcmd.push_back("-m");
                    wcmd.push_back(std::to_string(max_length));
                    wcmd.push_back("--profile");
                    wcmd.push_back(wprof.string());
                    wcmd.push_back(sdr_path.string());
                    (void)runCommandCapture(wcmd, wioutf, wierrf);
                }
            }

            if (run_compress) {
                const fs::path coutf = temp_dir / (name + "_compress_out.txt");
                const fs::path cerrf = temp_dir / (name + "_compress_err.txt");
                const auto r = runCommandCapture(
                    {ai_cli_path.string(), "-c", model_path.string(), format, sdr_path.string(), std::to_string(sparsity)},
                    coutf,
                    cerrf);
                row["compress"] = {
                    {"exit_code", r.exit_code},
                    {"elapsed_ms_wall", r.elapsed_ms},
                    {"command", r.command},
                    {"original_size_bytes", extractDouble(r.stdout_text, original_re)},
                    {"compressed_size_bytes", extractDouble(r.stdout_text, compressed_re)},
                    {"compression_ratio_reported", extractDouble(r.stdout_text, ratio_re)},
                    {"compression_time_reported_ms", extractDouble(r.stdout_text, ctime_re)},
                };
                row["compress"]["model_file_size_bytes"] = static_cast<double>(fs::file_size(model_path));
                if (fs::exists(sdr_path) && fs::is_regular_file(sdr_path)) {
                    const double sdr_file_size = static_cast<double>(fs::file_size(sdr_path));
                    row["compress"]["sdr_file_size_bytes"] = sdr_file_size;
                    row["sdr_size_bytes"] = sdr_file_size;
                    const auto orig = jsonNumberToDouble(row["compress"], "original_size_bytes")
                                          .value_or(static_cast<double>(fs::file_size(model_path)));
                    if (orig > 0.0) {
                        row["compress"]["compression_ratio_file_measured"] = orig / sdr_file_size;
                        row["compress"]["size_reduction_percent"] = (1.0 - (sdr_file_size / orig)) * 100.0;
                    }
                }
                {
                    const auto input_bytes = jsonNumberToDouble(row["compress"], "original_size_bytes")
                                                 .value_or(static_cast<double>(fs::file_size(model_path)));
                    const auto throughput = rateMBPerSec(input_bytes, r.elapsed_ms);
                    if (throughput.has_value()) {
                        row["compress"]["input_throughput_mb_per_sec"] = throughput.value();
                    }
                }
                if (r.exit_code != 0) {
                    row["status"] = "error";
                    row["error"] = "Compression failed";
                    row["compress"]["stderr"] = r.stderr_text;
                    run["results"].push_back(row);
                    continue;
                }
            }

            if (run_extract) {
                const fs::path eoutf = temp_dir / (name + "_extract_out.txt");
                const fs::path eerrf = temp_dir / (name + "_extract_err.txt");
                if (fs::exists(extract_dir)) {
                    fs::remove_all(extract_dir);
                }
                fs::create_directories(extract_dir);

                const auto r = runCommandCapture(
                    {ai_cli_path.string(), "-x", sdr_path.string(), extract_dir.string(), std::to_string(sparsity)},
                    eoutf,
                    eerrf);
                row["extract"] = {
                    {"exit_code", r.exit_code},
                    {"elapsed_ms_wall", r.elapsed_ms},
                    {"command", r.command},
                    {"extracted_size_bytes", directorySizeBytes(extract_dir)},
                };
                if (row["extract"]["extracted_size_bytes"].is_number()) {
                    const double ext_bytes = row["extract"]["extracted_size_bytes"].get<double>();
                    const auto throughput = rateMBPerSec(ext_bytes, r.elapsed_ms);
                    if (throughput.has_value()) {
                        row["extract"]["throughput_mb_per_sec"] = throughput.value();
                    }
                    if (row.contains("sdr_size_bytes") && row["sdr_size_bytes"].is_number()) {
                        const double sdr_size = row["sdr_size_bytes"].get<double>();
                        if (sdr_size > 0.0) {
                            row["extract"]["expansion_ratio_vs_sdr"] = ext_bytes / sdr_size;
                        }
                    }
                }
                if (r.exit_code != 0) {
                    row["status"] = "error";
                    row["error"] = "Extraction failed";
                    row["extract"]["stderr"] = r.stderr_text;
                    run["results"].push_back(row);
                    continue;
                }
            }

            if (run_inference) {
                const fs::path profile_json = temp_dir / (name + "_profile.json");
                const fs::path ioutf = temp_dir / (name + "_inference_out.txt");
                const fs::path ierrf = temp_dir / (name + "_inference_err.txt");
                if (fs::exists(profile_json)) {
                    fs::remove(profile_json);
                }

                std::vector<std::string> cmd = {text_cli_path.string()};
                if (mode == "legacy") {
                    cmd.push_back("--legacy");
                } else {
                    cmd.push_back("--on-demand");
                }
                cmd.push_back("-p");
                cmd.push_back(prompt);
                cmd.push_back("-m");
                cmd.push_back(std::to_string(max_length));
                cmd.push_back("--profile");
                cmd.push_back(profile_json.string());
                cmd.push_back(sdr_path.string());

                const auto r = runCommandCapture(cmd, ioutf, ierrf);
                json inf = {
                    {"exit_code", r.exit_code},
                    {"elapsed_ms_wall", r.elapsed_ms},
                    {"command", r.command},
                    {"mode", mode},
                    {"prompt_chars", prompt.size()},
                    {"prompt_token_estimate", estimateTokensSimple(prompt)},
                    {"max_length", max_length},
                };
                if (fs::exists(profile_json)) {
                    try {
                        const auto pj = json::parse(readTextFile(profile_json));
                        inf["profile"] = pj;
                        if (pj.contains("tokens_per_sec")) {
                            const double tps = pj["tokens_per_sec"].get<double>();
                            inf["tokens_per_min"] = tps * 60.0;
                        }
                        if (pj.contains("tokens") && pj["tokens"].is_number()) {
                            const double generated_tokens = pj["tokens"].get<double>();
                            inf["generated_tokens"] = generated_tokens;
                            if (r.elapsed_ms > 0.0) {
                                inf["generated_tokens_per_sec_wall"] = generated_tokens / (r.elapsed_ms / 1000.0);
                                inf["generated_tokens_per_min_wall"] = (generated_tokens / (r.elapsed_ms / 1000.0)) * 60.0;
                            }
                        }
                        if (pj.contains("duration_ms") && pj["duration_ms"].is_number()) {
                            const double engine_duration_ms = pj["duration_ms"].get<double>();
                            inf["engine_duration_ms"] = engine_duration_ms;
                            inf["wrapper_overhead_ms"] = r.elapsed_ms - engine_duration_ms;
                            if (engine_duration_ms > 0.0) {
                                inf["engine_overhead_ratio"] = r.elapsed_ms / engine_duration_ms;
                            }
                        }
                        if (pj.contains("tokens") && pj["tokens"].is_number() && pj.contains("duration_ms") && pj["duration_ms"].is_number()) {
                            const double t = pj["tokens"].get<double>();
                            const double dms = pj["duration_ms"].get<double>();
                            if (dms > 0.0) {
                                inf["generated_tokens_per_sec_engine"] = t / (dms / 1000.0);
                            }
                        }
                    } catch (...) {
                        inf["profile_parse_error"] = true;
                    }
                } else {
                    inf["profile_missing"] = true;
                }
                row["inference"] = inf;

                if (r.exit_code != 0) {
                    row["status"] = "error";
                    row["error"] = "Inference failed";
                    row["inference"]["stderr"] = r.stderr_text;
                }
            }

            double total_elapsed_ms = 0.0;
            if (row.contains("compress") && row["compress"].contains("elapsed_ms_wall") && row["compress"]["elapsed_ms_wall"].is_number()) {
                total_elapsed_ms += row["compress"]["elapsed_ms_wall"].get<double>();
            }
            if (row.contains("extract") && row["extract"].contains("elapsed_ms_wall") && row["extract"]["elapsed_ms_wall"].is_number()) {
                total_elapsed_ms += row["extract"]["elapsed_ms_wall"].get<double>();
            }
            if (row.contains("inference") && row["inference"].contains("elapsed_ms_wall") && row["inference"]["elapsed_ms_wall"].is_number()) {
                total_elapsed_ms += row["inference"]["elapsed_ms_wall"].get<double>();
            }
            row["total_elapsed_ms"] = total_elapsed_ms;
            run["results"].push_back(row);
        }
    }

    {
        std::vector<double> compress_ms;
        std::vector<double> extract_ms;
        std::vector<double> inference_ms;
        std::vector<double> tpm_wall;
        std::vector<double> compression_ratio;
        std::map<std::string, int> status_counts;

        for (const auto& row : run["results"]) {
            const std::string status = row.value("status", "unknown");
            status_counts[status]++;
            if (status != "ok") {
                continue;
            }
            if (row.contains("compress") && row["compress"].contains("elapsed_ms_wall") && row["compress"]["elapsed_ms_wall"].is_number()) {
                compress_ms.push_back(row["compress"]["elapsed_ms_wall"].get<double>());
            }
            if (row.contains("extract") && row["extract"].contains("elapsed_ms_wall") && row["extract"]["elapsed_ms_wall"].is_number()) {
                extract_ms.push_back(row["extract"]["elapsed_ms_wall"].get<double>());
            }
            if (row.contains("inference") && row["inference"].contains("elapsed_ms_wall") && row["inference"]["elapsed_ms_wall"].is_number()) {
                inference_ms.push_back(row["inference"]["elapsed_ms_wall"].get<double>());
            }
            if (row.contains("inference") && row["inference"].contains("generated_tokens_per_min_wall") &&
                row["inference"]["generated_tokens_per_min_wall"].is_number()) {
                tpm_wall.push_back(row["inference"]["generated_tokens_per_min_wall"].get<double>());
            }
            if (row.contains("compress") && row["compress"].contains("compression_ratio_file_measured") &&
                row["compress"]["compression_ratio_file_measured"].is_number()) {
                compression_ratio.push_back(row["compress"]["compression_ratio_file_measured"].get<double>());
            }
        }

        run["summary"] = {
            {"status_counts", status_counts},
            {"compress_elapsed_ms", summarizeSeries(compress_ms)},
            {"extract_elapsed_ms", summarizeSeries(extract_ms)},
            {"inference_elapsed_ms", summarizeSeries(inference_ms)},
            {"inference_generated_tpm_wall", summarizeSeries(tpm_wall)},
            {"compression_ratio_file_measured", summarizeSeries(compression_ratio)},
        };
    }

    {
        std::ofstream out(out_path);
        out << std::setw(2) << run << "\n";
    }

    std::cout << "Benchmark run complete: " << out_path.string() << "\n";
    std::cout << "Total tests: " << run["results"].size() << "\n";
    return 0;
}
