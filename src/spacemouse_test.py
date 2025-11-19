import ctypes

# Load libspnav
libspnav = ctypes.CDLL("libspnav.so")

# Event type constants
SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2

# Motion and button structs
class SpnavMotion(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("rz", ctypes.c_int)
    ]

class SpnavButton(ctypes.Structure):
    _fields_ = [
        ("bnum", ctypes.c_int),
        ("press", ctypes.c_int)
    ]

class SpnavEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("motion", SpnavMotion),
        ("button", SpnavButton)
    ]

# Connect to the daemon
if libspnav.spnav_open() == -1:
    raise RuntimeError("Could not connect to spacenavd. Is spacenavd running?")

print("Connected to spacenavd. Move the SpaceMouse... (Ctrl+C to quit)")

event = SpnavEvent()

while True:
    # spnav_poll_event returns >0 if an event is available
    if libspnav.spnav_poll_event(ctypes.byref(event)) > 0:
        if event.type == SPNAV_EVENT_MOTION:
            print(f"[MOTION] x={event.motion.x} y={event.motion.y} z={event.motion.z} "
                  f"rx={event.motion.rx} ry={event.motion.ry} rz={event.motion.rz}")

        elif event.type == SPNAV_EVENT_BUTTON:
            state = "pressed" if event.button.press else "released"
            print(f"[BUTTON] {event.button.bnum} {state}")
