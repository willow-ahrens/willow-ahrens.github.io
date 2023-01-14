mogrify -path thumbnails -format jpeg -auto-orient -thumbnail 180x500 -unsharp 0x.5 'images/*.jpeg'
mogrify -path thumbnails -format jpeg -auto-orient -thumbnail 180x500 -unsharp 0x.5 'images/*.jpg'
