from rle2mask import rle2mask
import numpy as np 
import matplotlib.pyplot as plt

def evaluemodel(vals, model, size=(256,256)):
    val = df_pneumo = vals[vals["EncodedPixels"] != ' -1']

        
    mask = rle2mask(val.values[3][1], 1024, 1024)
    mask = np.rot90(mask, 3) #rotating three times 90 to the right place
    mask = np.flip(mask, axis=1)
    mask = cv2.resize(mask, size)
    img = pydicom.read_file(val.values[3][-1]).pixel_array
    img = cv2.resize(img,size)
    

    fig = plt.figure(figsize=(15, 10))
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap='bone') #original x-ray
    a.set_title("Original x-ray image")
    plt.grid(False)
    plt.axis("off")
    

    a = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(mask, cmap='binary')
    a.set_title("The mask")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    a = fig.add_subplot(1, 3, 3)
    pred = model.predict(img)
    pred = np.squeeze(pred)
    plt.imshow(pred, cmap='binary', alpha=0.3)
    a.set_title("Mask on the x-ray: air in the pleura")
