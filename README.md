# Skipper

A desktop interaction tool that allows AI agents like Claude Code or OpenAI Codex to control web browsers and interact with desktop applications through natural language commands.

## Overview

Skipper enables AI agents to:
- View and navigate the web using your own browser
- Navigate to URLs
- Execute mouse clicks, keyboard input, and scrolling actions
- [SOON] Interact with any desktop application through natural language prompts

## The Vision
Skipper is a command line tool that gives AI agents the ability to interact with your browser/desktop. Unlike all-in-one computer use tools, Skipper takes the unix philosophy of "do one thing and do it well". Specifically, it is designed to be the "hands" of the AI agent, instead of the "brain".

Our architecture is designed to be privacy-preserving in the future. If the tool ends up being useful, we have designed the architecture to be possible to run locally only. The only sensitive information that would go to the cloud would be in text to the LLM agent, which could be censored or modified as necessary for privacy.

## Installation

### Prerequisites

- Python 3.9 or higher
- Chrome/Chromium browser with remote debugging enabled
- Gemini API key (for AI-powered interactions)
- Either
    - A computer capable of running OmniParser
    - An API key for Skipper to run this stage remotely

### Install Skipper

```bash
# Install
pip install skipper-tool

# (Optional) Install with local dependencies
pip install 'skipper-tool[local]'
```

### Setup Chrome Remote Debugging

1. Start Chrome with remote debugging enabled:
```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Linux
google-chrome --remote-debugging-port=9222

# Windows
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

2. Keep Chrome running in the background while using skipper_tool.

### Setup Initial Configuration

```bash
# Create a configuration file
skipper init --config

# This will prompt for your Gemini and Skipper API keys and create ~/.skipperrc
```

## Usage

Skipper provides three main commands for AI agents:

### 1. View Window State

```bash
skipper view
```

Returns the current state of the active browser window, including:
- Page title and URL
- A simplified HTML representation of the page

### 2. Navigate to URL

```bash
skipper navigate --url "https://example.com"
```

Navigates the browser to the specified URL and returns the new page state.

### 3. Execute Commands

```bash
skipper command --command_type <type> --prompt "<description>"
```

Execute actions on the current page:

- **Click**: `skipper command --command_type click --prompt "Click the login button"`
- **Type**: `skipper command --command_type type --prompt "Enter username: john.doe<Enter>"`
- **Scroll**: `skipper command --command_type scroll --prompt "Scroll down"`

## Integration with Claude Code

Skipper is designed to work seamlessly with Claude Code and other AI agents. To integrate it, simply usage information to your agent's configuration, such as `CLAUDE.md` or `AGENT.md`. For an example `AGENT.md` file, see [EXAMPLE.AGENT.md](EXAMPLE.AGENT.md).

## Advanced Features

### Debug Mode

Enable debug logging to save screenshots and detailed logs:

```bash
# Set debug folder in ~/.skipperrc
[debug]
enabled = true
folder = "/path/to/debug/folder"

# Or use environment variable
export SKIPPER_DEBUG_FOLDER="/path/to/debug/folder"
```

### Local AI Models

For enhanced privacy, you can use local AI models:

```bash
# Install local dependencies
pip install 'skipper-tool[local]'

# Configure local model paths in ~/.skipperrc
[models]
yolo_model_path = "/path/to/local/model.pt"
```

### Custom Configuration

Edit `~/.skipperrc` to customize:

```toml
[models]
screenshot_model = "gemini-2.5-flash"
ui_element_model = "gemini-2.5-pro"

[browser]
cdp_url = "http://localhost:9222"
context_index = 0
page_index = 0

[ui_interaction]
click_delay_seconds = 1.0
scroll_distance = 600
mouse_scale_factor = 0.5
```

## Troubleshooting

### Common Issues

1. **Chrome not responding**: Ensure Chrome is running with `--remote-debugging-port=9222`
2. **API key errors**: Set `GEMINI_API_KEY` environment variable or add to `~/.skipperrc`
3. **Permission errors**: Check that Skipper has access to the browser and debug port


```bash
# Enable verbose logging
export SKIPPER_DEBUG_FOLDER="/tmp/skipper-debug"
skipper view

# Check logs in the debug folder
ls /tmp/skipper-debug/
```

## Security Considerations

- Skipper requires access to your browser and can execute actions on your behalf
- API keys are stored locally in `~/.skipperrc`
- Debug mode saves screenshots locally - ensure the debug folder is secure
- Only use with trusted AI agents

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

AGPL

## Support

- Issues: [GitHub Issues](https://github.com/nharada1/skipper-tool/issues)