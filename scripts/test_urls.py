import requests

urls = {
    'tlc_yellow_2024_01': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
    'tlc_zone_lookup': 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv',
    'nyc_opendata_zones_geojson_1': 'https://data.cityofnewyork.us/api/geospatial/d3hk-qhbr?method=export&format=GeoJSON',
    'nyc_opendata_zones_geojson_2': 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Taxi_Zones/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson',
    'tlc_zone_shapefile_zip': 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
}

for name, url in urls.items():
    try:
        r = requests.head(url, timeout=10, allow_redirects=True, headers={'User-Agent': 'TrafficPulse/1.0'})
        ct = r.headers.get('Content-Type', 'Unknown')
        cl = r.headers.get('Content-Length')
        size_str = f"{int(cl)/(1024*1024):.2f} MB" if cl else "Unknown length"
        print(f"[{r.status_code}] {name}: {ct} ({size_str}) -> {url}")
    except Exception as e:
        print(f"[ERR] {name}: {e} -> {url}")
