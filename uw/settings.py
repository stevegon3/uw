import os, platform, socket, logging, yaml
import configparser

class Sett:
    def __init__(self, env='dev'):
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f) or {}
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        for handler in self.log.handlers[:]:
            self.log.removeHandler(handler)
        formatter = logging.Formatter(self.config['log_fmt'], datefmt=self.config['log_date_fmt'])
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.log.addHandler(ch)
        self.ini_file = self.config['ini_file']
        if os.name == 'nt':
            self.computer = 'Windows'
            self.ini_dir = r'S:\code\keys'
        elif socket.gethostname() == 'Steves-MBP' or socket.gethostname() == 'Mac' or platform.system() == 'Darwin':
            self.computer = 'Mac'
            self.ini_dir = '/Users/stevegon/SynologyDrive/code/keys'
        else:
            self.computer = 'ugali'
            self.ini_dir = '/home/steve/keys'
        config_ini = configparser.ConfigParser()
        ini_path = os.path.join(self.ini_dir, self.ini_file)
        config_ini.read(ini_path)

        if config_ini['pg']['PostgresHost'] is None or config_ini['pg']['PostgresHost'] == '':
            self.PostgresHost = 'ugali'
            self.PostgresDB = 'uw'
            self.PostgresPort = 5432
            self.PostgresUser = 'postgres'
            self.PostgresPassword = 'pg_secret'
        else:
            self.PostgresHost = config_ini['pg']['PostgresHost']
            self.PostgresDB = config_ini['pg']['PostgresDB']
            self.PostgresPort = config_ini['pg']['PostgresPort']
            self.PostgresUser = config_ini['pg']['PostgresUser']
            self.PostgresPassword = config_ini['pg']['PostgresPassword']
            self.PyInvDBPassword = config_ini['pg']['PyInvDBPassword']
        self.PostgresSchema = 'imt543'
        self.tz_east = self.config['tz_east']
        self.tz_pac = self.config['tz_pac']
        self.debug = self.config['debug']

def get_settings():
    return Sett()

sett = get_settings()