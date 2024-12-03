import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import streamlit as st

# Load the dataset at the beginning
df = pd.read_csv("pincode.csv", dtype={'Latitude': str, 'Longitude': str}, low_memory=False)

# Function to clean up and convert Latitude/Longitude
def convert_to_float(coord):
    if isinstance(coord, float):
        return coord
    
    coord = str(coord).strip().replace('-', '')
    if 'N' in coord or 'E' in coord:
        return float(coord.replace('N', '').replace('E', '').strip())
    elif 'S' in coord or 'W' in coord:
        return -float(coord.replace('S', '').replace('W', '').strip())
    else:
        try:
            return float(coord)
        except ValueError:
            print(f"Error converting {coord} to float. Check the data format.")
            return np.nan

df['Latitude'] = df['Latitude'].apply(convert_to_float)
df['Longitude'] = df['Longitude'].apply(convert_to_float)
df = df.dropna(subset=['Latitude', 'Longitude'])
coordinates = df[['Latitude', 'Longitude']].values

n_centroids = 35  
kmeans = KMeans(n_clusters=n_centroids, init='k-means++', n_init=20, max_iter=500, random_state=42)
kmeans.fit(coordinates)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

df['Cluster'] = cluster_labels

unique_centroids = np.unique(centroids, axis=0)


def find_nearest_service_centroid(centroid, df):
    distances = cdist([centroid], df[['Latitude', 'Longitude']])
    nearest_index = distances.argmin()
    return df.iloc[nearest_index]

nearest_service_centers = [find_nearest_service_centroid(centroid, df) for centroid in centroids]

def get_nearest_service_center(pincode_input):
    try:
        user_location = df[df['Pincode'] == pincode_input]
        if user_location.empty:
            return None

        user_latlon = user_location[['Latitude', 'Longitude']].values[0]
        distances = cdist([user_latlon], centroids)
        nearest_centroid_idx = distances.argmin()
        nearest_center = nearest_service_centers[nearest_centroid_idx]
        return nearest_center

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

st.title("Service Management and Automation Tool")
user_pincode = st.text_input("Enter your Pincode:", "")

if st.button("Find Nearest Service Center"):
    if user_pincode:
        try:
            user_pincode = int(user_pincode)
            nearest_center = get_nearest_service_center(user_pincode)

            if nearest_center is not None:
                st.success(f"Nearest Service Center for Pincode {user_pincode}:")
                st.write(f"   Pincode: {nearest_center['Pincode']}")
                st.write(f"   Circle Name: {nearest_center['CircleName']}")
                st.write(f"   State Name: {nearest_center['StateName']}")
                st.write(f"   Contact Number: {nearest_center['Contact Number']}")
                st.write(f"   Latitude: {nearest_center['Latitude']:.4f}")
                st.write(f"   Longitude: {nearest_center['Longitude']:.4f}")

                # Prepare the data for the map
                map_data = pd.DataFrame({
                    'lat': [nearest_center['Latitude']],
                    'lon': [nearest_center['Longitude']]
                })
                st.map(map_data)  # Display the map with the nearest service center
                
                # Create a Google Maps link
                google_maps_url = f"https://www.google.com/maps/search/?api=1&query={nearest_center['Latitude']},{nearest_center['Longitude']}"
                st.markdown(f"[Open in Google Maps]({google_maps_url})", unsafe_allow_html=True)
            else:
                st.warning("No service center found.")
        except ValueError:
            st.error("Please enter a valid pincode.")

