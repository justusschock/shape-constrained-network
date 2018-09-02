import yaml


class NetConfig(object):
    """
    Implements parser for configuration files
    """

    def __init__(self, config, user: str, verbose=True):
        """

        Parameters
        ----------
        config: string
            configuration file
        user: string
            user to parse inside the file
        verbose: bool
            verbosity
        """
        with open(config, 'r') as stream:
            docs = yaml.load_all(stream)
            user_found = False
            for doc in docs:
                for k, v in doc.items():
                    if k == user:
                        user_found = True
                        self._state_dict = {}
                        for k1, v1 in v.items():

                            cmd = k1 + "=" + repr(v1)
                            if verbose:
                                print(cmd)
                            cmd = "self." + cmd
                            exec(cmd)
                            self._state_dict[k1] = v1

        if not user_found:
            raise ValueError("Unknown User %s" % user)

    def to_dict(self):
        """
        Returns class  attributes as dict

        Returns
        -------
        dict: parsed arguments
        """
        return self._state_dict

    @classmethod
    def dict_from_yaml(cls, config, user: str, verbose=True):
        """
        Creates class instance and returns dict

        Parameters
        ----------
        config: string
            configuration file
        user: string
            user to parse inside the file
        verbose: bool
            verbosity

        Returns
        -------
        dict: parsed arguments
        """
        return cls(config, user, verbose).to_dict()
