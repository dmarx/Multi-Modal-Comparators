"""
This file contains descriptors for generic data modalities that a particular MMC may support.

The purpose of these classes is to facilitate determining modality compatibility between MMCs, 
and to abstract away specifics of data management that should be common to all media of a given modality.
"""

# for minimization of dependencies, maybe we could move this import inside the modalities that need it?
# e.g. PIL wouldn't get imported until an Image modality object is invoked somewhere in the library.
import PIL

class Modality:
    """
    A "modality" in our context is a distribution over data that is limited to a single medium. 
    This class characterizes modalities via metadata attributes and serialization methods.
    """
    def __init__(self, 
                name,
                #read_func,
                #write_func,
                #default_loss,
                #default_projector, # e.g. =CLIP,
                ):
        self.name=name

    def read_from_disk(self, fpath):
        """
        Preferred method for loading data of this modality from disk
        """
        with open(fpath, 'r') as f:
            return f.read()

    def write_to_disk(self, fpath, obj):
        """
        Preferred method for writing data of this modality from disk
        """
        with open(fpath, 'w') as f:
            return f.write(obj)


# to do: better mechanism for registering modalities

TEXT = Modality(name='text')

IMAGE = Modality(name='image')
IMAGE.read_from_disk = PIL.Image.open
IMAGE.write_to_disk = lambda fpath, obj: obj.save(fpath)

AUDIO = Modality('audio')
