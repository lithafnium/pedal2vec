import pedalboard
from threading import Event, Thread

plugin = pedalboard.load_plugin("/Library/Audio/Plug-Ins/Components/SmartPedal.component")
close_window_event = Event()
print(plugin)
should_close_window = False
def other_thread():
    # do something to determine when to close the window
    if should_close_window:
        close_window_event.set()

thread = Thread(target=other_thread)
thread.run()

# This will block until the other thread calls .set():
plugin.show_editor(close_window_event)



from pedalboard import Pedalboard, Chorus, Compressor, Delay, Gain, Reverb, Phaser
from pedalboard.io import AudioStream

# input_device_name = "MacBook Pro Microphone"
input_device_name = "Scarlett Solo USB"
output_device_name = "External Headphones"

print("Loading stream...")

with AudioStream(input_device_name, output_device_name) as stream:
    # In this block, audio is streaming through `stream`!
    # Audio will be coming out of your speakers at this point.
    reverb = Reverb()
    # Add plugins to the live audio stream:
    stream.plugins = Pedalboard([reverb])

    # Change plugin properties as the stream is running:
    # reverb.wet_level = 1.0
    input("listneing to audio....")
    # Delete plugins:
    del stream.plugins[0]


# from pedalboard import Pedalboard, Chorus, Compressor, Delay, Gain, Reverb, Phaser
# from pedalboard.io import AudioStream

# print("opening stream...")
# # Open up an audio stream:
# with AudioStream(
#   input_device_name="Scarlett Solo USB",  # Guitar interface
#   output_device_name="MacBook Pro Speakers",
#   buffer_size=512
# ) as stream:
#   # Audio is now streaming through this pedalboard and out of your speakers!
#   stream.plugins = Pedalboard([
#      plugin
#     #   Compressor(threshold_db=-50, ratio=25),
#     #   Gain(gain_db=30),
#     #   Chorus(),
#     #   Phaser(),
#     #   Reverb(room_size=0.25),
#   ])
#   input("Press enter to stop streaming...")