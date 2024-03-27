from utils.config import Config

cfg = Config.get_config()

cfg_text = Config.pretty_text(cfg)
print(cfg_text)
