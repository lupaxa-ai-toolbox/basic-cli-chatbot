<p align="center">
  <a href="https://github.com/lupaxa-ai-toolbox">
    <img src="https://raw.githubusercontent.com/the-lupaxa-project/brand-assets/master/logos/organisations/ai-toolbox/readme-logo.png" alt="AI Toolbox" />
  </a>
</p>

<h1 align="center">Basic CLI Chatbot</h1>

A command-line chatbot that talks to OpenAI, Gemini, Grok, or a local Ollama
server. A chat requires a provider and a model.

The PyPI name is `lupaxa-basic-cli-chatbot`. The console script is
`basic-cli-chatbot`.

## Install

Python 3.10 or newer. Set `OPENAI_API_KEY` for `--provider openai`,
`GEMINI_API_KEY` for `--provider gemini`, and `XAI_API_KEY` for
`--provider grok`. Ollama uses a local server and does not need those keys.

```bash
pip install lupaxa-basic-cli-chatbot
```

## Homebrew

```bash
brew tap the-lupaxa-project/tap
brew trust the-lupaxa-project/tap
brew install basic-cli-chatbot
```

## Usage

A command with no question opens a prompt. A question after the flags asks
once and exits.

```bash
basic-cli-chatbot --provider openai --model gpt-5.6-luna
basic-cli-chatbot --provider openai --model gpt-5.6-luna "What is 2+2?"
basic-cli-chatbot --provider ollama --model llama3.2
basic-cli-chatbot --provider grok --model grok-4.6
basic-cli-chatbot --list-models --provider ollama
basic-cli-chatbot --show-config --provider gemini
basic-cli-chatbot --version
basic-cli-chatbot --help
```

`--provider` overrides `BASIC_CLI_CHATBOT_PROVIDER`. `--model` overrides
`BASIC_CLI_CHATBOT_MODEL`.

Ollama talks to a server at `http://localhost:11434`. That server has to be
running, and the model has to be installed already (`ollama pull llama3.2`).

A YAML profile file can supply the provider, the model, and the optional
chat settings. The default path is `$XDG_CONFIG_HOME/basic-cli-chatbot/config.yaml`,
or `~/.config/basic-cli-chatbot/config.yaml` when `XDG_CONFIG_HOME` is unset.
`--config` points at another file.

```yaml
default:
  provider: openai
  model: gpt-5.6-luna
code-review:
  provider: ollama
  model: llama3.2
  ollama-host: http://127.0.0.1:11434
  timeout: 120
```

`--profile code-review` uses that profile and cannot be combined with
`--provider` or `--model`. With neither of those flags and no `--profile`,
a `default` profile is used. `--ollama-host`, `--max-context-messages`,
`--timeout`, `--stream`, and `--no-stream` replace the profile value when
you pass them. `--validate` checks the file and exits. API keys stay in
the environment.

`--ollama-host` overrides `http://localhost:11434`.

`--max-context-messages` defaults to `0`, which keeps the whole
conversation.

Replies print as they arrive. `--no-stream` prints the reply in one piece.

The reply wait is 60 seconds for OpenAI, Gemini, and Grok, and 300 seconds
for Ollama. `--timeout` overrides that. `BASIC_CLI_CHATBOT_TIMEOUT` is used
only when `--timeout` is unset and the profile has no `timeout`.

`exit` and `quit` stop the prompt. `clear` drops the conversation.
`/context` prints how many messages are kept. `/provider` and `/model`
print the current names.

`--show-config` reports a key as `configured` or `not configured` and
does not print the secret.

## Development

```bash
make init
make python-install-dev
make python-check
```

<a href="https://github.com/the-lupaxa-project">
    <img src="https://raw.githubusercontent.com/the-lupaxa-project/brand-assets/master/logos/components/footer-for-child-orgs.svg" alt="The Lupaxa Project Footer" width="100%" />
</a>
