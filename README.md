# Image Selection
```
This service is an implementation of a paper "Thumbnail Image Selection for VOD Services".
https://ieeexplore.ieee.org/document/8695315

``` 

## TODO
```.env
- The status of the project is in development.
- The project will be refactored in the future(maybe...lol).
- You can still run this service to get the best images.

```
## Environment Variable
```
- Required
  - ENV : dev [dev|prod]
  - TYPE: is [is]
  - SECRET_FILE: ./config/prodis

Required configs in SECRET_FILE(json):

{
        "video_id": "love_actually",
        "video_download_dir":"./video_cache",
	    "clean_folder": "False",
	    "video_start": "0.10",
	    "video_end": "0.90",
	    "animation": 0,
	    "frame_diff_threshold": 90,
	    "sharpness_threshold": 90,
	    "sharpness_diff_threshold": 60000,
	    "output_images": 5,
	    "subtitled": "True"
}
In this case:
Video path should be ./video_cache/love_actually/love_actually.mp4 
"video_start": "0.10" => Processing video starts at 10% of the video
"video_end": "0.90" => Processing video ends at 90% of the video
"output_images": 5, => The best 5 selected images 

```

## Usage
```.env
 python start_worker.py 
 The selected images will be output in ./video_cache/love_actually/ folder
```