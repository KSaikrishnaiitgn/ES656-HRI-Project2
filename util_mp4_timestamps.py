"""Read creation time and duration from the 'mvhd' atom of MP4 files (no ffprobe needed).
MP4 times are seconds since 1904-01-01 UTC. Also prints any '©xyz'/date-like udta strings."""
import struct, sys, datetime, os

EPOCH = datetime.datetime(1904, 1, 1, tzinfo=datetime.timezone.utc)


def find_atoms(f, start, end, path=()):
    f.seek(start)
    while f.tell() < end:
        pos = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack('>I4s', hdr)
        hl = 8
        if size == 1:
            size = struct.unpack('>Q', f.read(8))[0]; hl = 16
        elif size == 0:
            size = end - pos
        typ = typ.decode('latin1')
        yield pos, size, hl, path + (typ,)
        if typ in ('moov', 'trak', 'mdia', 'udta', 'meta'):
            inner = pos + hl + (4 if typ == 'meta' else 0)
            yield from find_atoms(f, inner, pos + size, path + (typ,))
        f.seek(pos + size)


for p in sys.argv[1:]:
    with open(p, 'rb') as f:
        end = os.path.getsize(p)
        for pos, size, hl, path in find_atoms(f, 0, end):
            if path[-1] == 'mvhd':
                f.seek(pos + hl)
                ver = f.read(1)[0]; f.read(3)
                if ver == 1:
                    ct, mt, ts, dur = struct.unpack('>QQIQ', f.read(28))
                else:
                    ct, mt, ts, dur = struct.unpack('>IIII', f.read(16))
                c = EPOCH + datetime.timedelta(seconds=ct)
                m = EPOCH + datetime.timedelta(seconds=mt)
                ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
                print(f"{os.path.basename(p)}: created {c.astimezone(ist):%Y-%m-%d %H:%M:%S} IST, "
                      f"modified {m.astimezone(ist):%H:%M:%S} IST, duration {dur/ts:.2f} s")
            if path[-1] in ('\xa9xyz', '\xa9day', 'date'):
                f.seek(pos + hl); print('   ', path[-1], f.read(min(size - hl, 64)))
