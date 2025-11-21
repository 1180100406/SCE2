import configparser

config = configparser.ConfigParser()
config.optionxform = str
config.read('/home/yuanwenyu/DCA-HRL_explore/algo/drnd/config.conf')

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]
