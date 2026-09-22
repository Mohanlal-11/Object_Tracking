import torch

def sort_images(shapes, out, dim):
    shapes.sort(key=lambda x: x[dim])
    out.append(shapes.pop()[2])
    if dim == 0:
        out.append(shapes.pop()[2])
        out.append(shapes.pop(1)[2])
    else:
        out.append(shapes.pop(1)[2])
        out.append(shapes.pop()[2])
    out.append(shapes.pop(0)[2])
    if shapes:
        sort_images(shapes, out, dim)
        
def mosaic_augment(images, targets, cls_labels,ch):
    assert len(images) % 4 == 0, "mosaic augmentation: len(images) % 4 != 0"
    shapes = [(img.shape[1], img.shape[2], i) for i, img in enumerate(images)]
    ratios = [int(h >= w) for h, w, _ in shapes]
    dim = int(sum(ratios) >= len(ratios) * 0.5)
    order = []
    sort_images(shapes, order, dim)
    
    new_images, new_targets, new_labels = [], [], []
    for i in range(len(order) // 4):
        hs, ws = zip(*[images[o].shape[-2:] for o in order[4 * i:4 * (i + 1)]])
        tl_y, br_y = max(hs[0], hs[1]), max(hs[2], hs[3])
        tl_x, br_x = max(ws[0], ws[2]), max(ws[1], ws[3])
        merged_image = images[0].new_zeros((ch, tl_y + br_y, tl_x + br_x))

        Aug_targets = []
        for j in range(4):
            index = order[4 * i + j]
            img = images[index]
            box = targets[index]
            h, w = img.shape[-2:]

            x1, y1, x2, y2 = tl_x, tl_y, tl_x, tl_y
            if j == 0: # top left
                x1 -= w
                y1 -= h
            elif j == 1: # top right
                x2 += w
                y1 -= h
            elif j == 2: # bottom left
                x1 -= w
                y2 += h
            elif j == 3: # bottom right
                x2 += w
                y2 += h

            merged_image[:, y1:y2, x1:x2].copy_(img)
            box[:, [0, 2]] += x1
            box[:, [1, 3]] += y1
            
            targets[index] = box
            
        boxes = torch.cat([targets[o] for o in order[4 * i:4 * (i + 1)]])
        labels = torch.cat([cls_labels[o] for o in order[4 * i:4 * (i + 1)]])
        
        new_images.append(merged_image)
        new_targets.append(boxes)
        new_labels.append(labels)
    return new_images, new_targets, new_labels