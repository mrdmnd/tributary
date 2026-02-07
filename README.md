# tributary
A foundation model transformer for Relational Databases


## Compiling Flash Attention
Flash attention takes a notoriously long time to compile (30m+).

Do this once and cache it so you don't have to do it again.

```bash
# Create the directory
mkdir -p ~/.cache/candle-flash-attn


# In ~/.bashrc or ~/.zshrc
export CANDLE_FLASH_ATTN_BUILD_DIR="$HOME/.cache/candle-flash-attn"

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```
