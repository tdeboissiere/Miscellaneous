# Miscellaneous tips

## Image operations

### Resize image
	convert -geometry 800x800 -density 200x200 -quality 100 image_in image_out

### Convert to black and white
	convert image_int -monochrome image_out (black and white)

## Synchronize directories
	rsync -avz local_dir remote_dir 