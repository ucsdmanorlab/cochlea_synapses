# when new point is added:
if viewer.dims.ndisplay == 3:
    count = 0
    for pt in pts.data:
        idx = tuple(pt.astype(int))
        maxz = np.argmax(img.data[:, idx[1], idx[2]])
        pts.data[count, 0] = maxz
        count += 1

