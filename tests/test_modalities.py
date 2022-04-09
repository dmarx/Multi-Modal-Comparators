
def test_TEXT():
    from mmc.modalities import TEXT
    TEXT.name == 'text'

def test_IMAGE():
    from mmc.modalities import IMAGE
    IMAGE.name == 'image'


def test_AUDIO():
    from mmc.modalities import AUDIO
    AUDIO.name == 'audio'


def test_Modality():
    from mmc.modalities import Modality
    FOO = Modality(name='foo')
    FOO.name ==  'foo'

#TEXT = Modality(name='text')


#def test_():
#    from mmc.modalities import


#def test_():
#    from mmc.modalities import


#def test_():
#    from mmc.modalities import


#read_from_disk
#write_to_disk
#TEXT = Modality(name='text')

#IMAGE = Modality(name='image')
#IMAGE.read_from_disk = PIL.Image.open
#IMAGE.write_to_disk = lambda fpath, obj: obj.save(fpath)

#AUDIO = Modality('audio')
