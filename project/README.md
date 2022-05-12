# Jetson Weather Forecast

Run the project in order the below. After the cache file is made, no other input is required and you can run as many forecasts as you want until you want to train again.

## Build
mkdir build && cd build && cmake ../ && sudo make install && sudo ldconfig

## Generate Features
Takes a directory full of pictures, and featurizes them into a single json file. While in the pictures directory, call like this. The JSON file MUST be in a different location than the directory with the training pictures or else the C++ filesystem module will try to read the JSON file as a picture.

ex. generate_features /tmp/sunny.json $(pwd) sunny

## Generate Cache
Takes a JSON file full of features and condenses them into a single feature in a cache file. After adding all feature lists to the cache file, you MUST manually edit the cache file to be a JSON array. The array format was not implemented in code due to time constraints. Brackets must be added to the front and back of the cache features, and a comma put between them.

ex. generate_cache /tmp/sunny.json /tmp/cache.json && generate_cache /tmp/cloudy.json /tmp/cache.json

## Weather Forecast
Takes a picture and computes which element of the cache file matches the closest through a minimum distance calculation. The entry specified by the output is the weather classification.

ex. weather_forecast jforecast_14463_s00_00000.jpg /tmp/cache.json
