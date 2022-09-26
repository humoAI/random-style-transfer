import torchfile
class pytorch_lua_wrapper:
    def __init__(self, lua_path):
        self.lua_model = torchfile.load(lua_path)

    def get(self, idx):
        return self.lua_model._obj.modules[idx]._obj

