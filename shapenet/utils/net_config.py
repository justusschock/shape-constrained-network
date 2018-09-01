import yaml
import inspect


class NetConfig(object):

    def __init__(self, config, user: str, verbose=True):
        with open(config, 'r') as stream:
            docs = yaml.load_all(stream)
            user_found = False
            for doc in docs:
                for k, v in doc.items():
                    if k == user:
                        user_found = True
                        self._state_dict = {}
                        for k1, v1 in v.items():
                            # allow 1:1 execution of python code in the yaml file
                            if k1 == "verbatim_block":
                                if verbose:
                                    print ("===START VERBATIM BLOCK===\n" + v1 + "\n===END VERBATIM BLOCK===")
                                exec(v1)
                            else:
                                cmd = k1 + "=" + repr(v1)
                                if verbose:
                                    print(cmd)
                                cmd = "self." + cmd
                                exec(cmd)
                                self._state_dict[k1] = v1

        if not user_found:
            raise ValueError("Unknown User %s" % user)

    def to_dict(self):
        return self._state_dict

    @classmethod
    def dict_from_yaml(cls, config, user: str, verbose=True):
        return cls(config, user, verbose).to_dict()