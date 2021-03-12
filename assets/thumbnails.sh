mogrify -path thumbs -format gif -define jpeg:size=180x500 -auto-orient -thumbnail 90x250 -unsharp 0x.5  'images/*.jpg'
