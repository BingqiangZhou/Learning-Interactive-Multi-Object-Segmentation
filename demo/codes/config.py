class GlobalConfig(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
            return cls._instance
        return cls._instance
    
    def __init__(self) -> None:
        pass
    
    def init_args(self, args):
        self._init_args = args.copy()
        self._args = args.copy()
    
    def restore_arg(self):
        self.args = self._init_args
        return self._init_args
    
    @property
    def args(self):
        return self._args
    
    @args.setter
    def args(self, value):
        if self._args is None:
            self._args = value


    