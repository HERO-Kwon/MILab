from google.cloud import vision
key='AIzaSyCXahk2MpWVzqExMB-bMOBUFlXA0dCYf-0'

global GOOGLE_APPLICATION_CREDENTIALS
GOOGLE_APPLICATION_CREDENTIALS='C:\\Users\\herok\\OneDrive\\1_MI_Lab\\Work\\DetectFaces\\MI-DetectFaces-dfb6473c45c8.json'


client = vision.ImageAnnotatorClient(credentials=GOOGLE_APPLICATION_CREDENTIALS)


def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient(credentials=GOOGLE_APPLICATION_CREDENTIALS)

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(image=image).face_annotations

    
def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')

    im.save(output_filename)


import PIL
import base64


file = 'D:\\Matlab_Drive\\Data\\gsearch_small_image\\download (18).jpg'
with open(file,"rb") as image_file:
    image = base64.b64encode(image_file.read())


faces = detect_face(img,3)

#    with open(input_filename, 'rb') as image:
faces = detect_face(image, max_results)
print('Found {} face{}'.format(
    len(faces), '' if len(faces) == 1 else 's'))

print('Writing to file {}'.format(output_filename))
# Reset the file pointer, so we can read the file again
image.seek(0)
highlight_faces(image, faces, output_filename)