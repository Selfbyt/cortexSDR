const https = require('https');
const fs = require('fs');
const path = require('path');

const images = [
  {
    url: 'https://www.performanceplustire.com/wp-content/uploads/2023/01/hero-bg.jpg',
    filename: 'hero-bg.jpg'
  },
  {
    url: 'https://www.performanceplustire.com/wp-content/uploads/2023/01/tire-services.jpg',
    filename: 'tire-services.jpg'
  },
  {
    url: 'https://www.performanceplustire.com/wp-content/uploads/2023/01/auto-repair.jpg',
    filename: 'auto-repair.jpg'
  },
  {
    url: 'https://www.performanceplustire.com/wp-content/uploads/2023/01/maintenance.jpg',
    filename: 'maintenance.jpg'
  },
  {
    url: 'https://www.performanceplustire.com/wp-content/uploads/2023/01/logo.png',
    filename: 'logo.png'
  }
];

const publicDir = path.join(__dirname, '..', 'public', 'images');

// Create the images directory if it doesn't exist
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir, { recursive: true });
}

images.forEach(image => {
  const filePath = path.join(publicDir, image.filename);
  const file = fs.createWriteStream(filePath);

  https.get(image.url, response => {
    response.pipe(file);

    file.on('finish', () => {
      file.close();
      console.log(`Downloaded ${image.filename}`);
    });
  }).on('error', err => {
    fs.unlink(filePath, () => {}); // Delete the file if there's an error
    console.error(`Error downloading ${image.filename}:`, err.message);
  });
}); 