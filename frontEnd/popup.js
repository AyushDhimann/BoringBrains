document.addEventListener('DOMContentLoaded', () => {
  // Page elements
  const pages = {
    login: document.getElementById('login-page'),
    control: document.getElementById('control-page'),
    results: document.getElementById('results-page')
  };

  // UI Elements
  const loginButton = document.getElementById('login-button');
  const logoutButtons = document.querySelectorAll('#logout-button, #results-logout-button');
  const screenshotToggle = document.getElementById('screenshot-toggle');
  const queryButton = document.getElementById('query-button');
  const backButton = document.getElementById('back-button');
  const errorMessage = document.getElementById('error-message');

  // Initialize UI based on session
  chrome.storage.local.get(['session', 'screenshotEnabled'], (result) => {
    if (result.session) {
      showPage('control');
      screenshotToggle.checked = result.screenshotEnabled || false;
    }
  });

  // Login Handler
  loginButton.addEventListener('click', async () => {
    const serverIp = document.getElementById('server-ip').value.trim();
    const token = document.getElementById('token').value.trim();

    if (!serverIp || !token) {
      errorMessage.textContent = 'Please fill in all fields';
      return;
    }

    try {
      await sendLoginRequest(serverIp, token);
      chrome.storage.local.set({ session: { serverIp, token } });
      showPage('control');
    } catch (error) {
      errorMessage.textContent = error.message;
    }
  });

  // Logout Handler
  logoutButtons.forEach(button => {
    button.addEventListener('click', () => {
      chrome.storage.local.remove(['session', 'screenshotEnabled']);
      chrome.runtime.sendMessage({ action: 'stopScreenshots' });
      showPage('login');
    });
  });

  // Screenshot Toggle Handler
  screenshotToggle.addEventListener('change', () => {
    const isEnabled = screenshotToggle.checked;
    chrome.storage.local.set({ screenshotEnabled: isEnabled });
    chrome.runtime.sendMessage({
      action: isEnabled ? 'startScreenshots' : 'stopScreenshots'
    });
  });

  // Query Handler
  queryButton.addEventListener('click', async () => {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;

    try {
      const response = await sendQuery(query);
      renderResults(query, response);
      showPage('results');
    } catch (error) {
      console.error('Query error:', error);
    }
  });

  // Back Button Handler
  backButton.addEventListener('click', () => {
    showPage('control');
  });

  // Helper Functions
  async function sendLoginRequest(serverIp, token) {
    const deviceId = crypto.randomUUID();
    const response = await fetch(`http://${serverIp}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        device_id: deviceId,
        token,
        device_name: 'chrome extension'
      })
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    return response.json();
  }

  async function sendQuery(query) {
    const response = await fetch('http://1725364.xyz:9000/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });

    if (!response.ok) {
      throw new Error('Query failed');
    }

    return response.json();
  }

  function renderResults(query, data) {
    const queryDisplay = document.querySelector('.query-display');
    const answerText = document.querySelector('.answer-text');
    const imagesContainer = document.querySelector('.results-images');

    queryDisplay.textContent = query;
    answerText.textContent = data.answer || 'No answer available';

    // Clear previous images
    imagesContainer.innerHTML = '';

    // Handle images
    if (data.relevant_images && data.relevant_images.length > 0) {
      // Change to flex container for vertical stacking
      imagesContainer.style.display = 'flex';
      imagesContainer.style.flexDirection = 'column';
      imagesContainer.style.gap = '16px';

      data.relevant_images.forEach(imageInfo => {
        try {
          // Create image container
          const imgWrapper = document.createElement('div');
          imgWrapper.className = 'image-wrapper';

          // Create image element
          const img = document.createElement('img');
          img.className = 'result-image';

          // Use the data property from the image info
          if (imageInfo.data) {
            const imageData = imageInfo.data.startsWith('data:image')
              ? imageInfo.data
              : `data:image/jpeg;base64,${imageInfo.data}`;

            img.src = imageData;
            img.alt = `Image from ${imageInfo.path || 'search result'}`;

            // Add load event listener to handle resizing after image loads
            img.addEventListener('load', () => {
              imgWrapper.appendChild(img);
              // Update extension height after each image loads
              updateExtensionHeight();
            });

            // Add error handling
            img.addEventListener('error', (e) => {
              console.error('Failed to load image:', imageInfo.path);
              imgWrapper.innerHTML = `<div class="error-message">Failed to load image: ${imageInfo.path}</div>`;
              updateExtensionHeight();
            });

            imagesContainer.appendChild(imgWrapper);
          }
        } catch (error) {
          console.error('Error processing image:', error);
        }
      });
    } else {
      imagesContainer.innerHTML = '<div class="no-images">No relevant images found</div>';
    }

    // Add or update styles
    if (!document.querySelector('#image-styles')) {
      const style = document.createElement('style');
      style.id = 'image-styles';
      style.textContent = `
        .results-images {
          width: 100%;
          padding: 0;
          margin: 16px 0;
        }
        .image-wrapper {
          width: 100%;
          margin: 0;
        }
        .result-image {
          width: 100%;
          height: auto;
          display: block;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .error-message {
          color: red;
          padding: 10px;
          background: #ffebee;
          border-radius: 4px;
          margin: 8px 0;
        }
        .no-images {
          padding: 10px;
          color: #666;
          text-align: center;
          margin: 8px 0;
        }
      `;
      document.head.appendChild(style);
    }

    // Initial height update
    updateExtensionHeight();
  }

  // Separate function to update extension height
  function updateExtensionHeight() {
    // Get the actual content height
    const contentHeight = document.body.scrollHeight;

    // Add some padding to the bottom
    const padding = 32;

    // Set minimum height
    const minHeight = 500;

    // Calculate new height
    const newHeight = Math.max(contentHeight + padding, minHeight);

    // Update body height
    document.body.style.height = `${newHeight}px`;

    // Update popup dimensions
    if (chrome.windows && chrome.windows.update) {
      chrome.windows.getCurrent((window) => {
        chrome.windows.update(window.id, {
          height: newHeight
        });
      });
    }
  }

  function showPage(pageName) {
    Object.values(pages).forEach(page => {
      page.classList.remove('active');
    });
    pages[pageName].classList.add('active');

    // Reset scroll position
    window.scrollTo(0, 0);
  }
});
