# Boring Brains - Bb - Your AI-Powered Digital Recall Assistant

Bb is a Chrome extension designed to help you search and recall your digital history effortlessly. Built with privacy in mind, Bb runs exclusively on open-source models powered by SambaNova and is fully self-hosted, ensuring complete control over your data.

Whether you're studying, shopping, working, or simply browsing, Bb keeps track of the information you may need later. With its AI capabilities, it quickly retrieves the content you're looking for, making forgotten details a thing of the past.

## Features

- **Privacy-Focused**: All data is processed locally on your device or server.
- **AI-Powered Search**: Quickly retrieve browsing history, screenshots, and forgotten content.
- **Easy Setup**: Host on your local machine or a remote web server.
- **Chrome Integration**: Directly integrates with Chrome to enable seamless searching.

## Installation (Linux)

### Prerequisites

Make sure your system has the following installed:

- Docker Compose
- `ffmpeg`, `libsm6`, and `libxext6`

### Steps to Install

1. Install dependencies:
   ```bash
   sudo apt install docker-compose ffmpeg libsm6 libxext6 -y
   ```

2. Clone the Bb repository from GitHub:
   ```bash
   git clone https://github.com/AyushDhimann/BoringBrains
   ```

3. Navigate to the backend directory:
   ```bash
   cd BoringBrains/backEnd
   ```

4. Copy the environment file and add your SAMBANOVA_API_KEY:
   ```bash
   cp .env.example .env
   ```

5. Build the Docker containers:
   ```bash
   docker-compose build
   ```

6. Start the server:
   ```bash
   docker-compose up
   ```

   - Wait for the server to initialize.
   - A one-time login token will be generated. **Save this token safely.**

7. Configure the Chrome extension:
   - Open the Bb Chrome extension.
   - Paste the login token and enter your server's IP address.
   - If you're hosting locally, use your local IP or `localhost`.

8. Allow the tool to take screenshots when you wish.

### Usage

- Enable screenshot permissions for your Chrome browser.
- Type queries into the Bb Chrome extension to recall past browsing sessions.
- Search for specific items or content, and Bb will retrieve the associated screenshots and data within seconds.

## AI Features

Bb leverages Langchain's agent-based and prompt chaining techniques to deliver a production-level AI-powered experience. Its core functionality is built on advanced AI models for efficient search and recall.

### Forget the past—Bb is here to remember it for you.
