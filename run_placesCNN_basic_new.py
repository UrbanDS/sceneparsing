# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import glob

from database import mydatabase
dbms = mydatabase.MyDatabase(mydatabase.SQLITE, dbname='/Users/divyachandana/Documents/NJIT/work/summertasks/may25-may30/Park_face/mydb.sqlite')
# th architecture to use
arch = 'resnet18'
places_365_dict = {'airfield': 'outdoor', 'airplane_cabin': 'indoor', 'airport_terminal': 'outdoor', 'alcove': 'outdoor', 'alley': 'outdoor', 'amphitheater': 'outdoor', 'amusement_arcade': 'indoor', 'amusement_park': 'outdoor', 'apartment_building/outdoor': 'outdoor', 'aquarium': 'indoor', 'aqueduct': 'outdoor', 'arcade': 'indoor', 'arch': 'outdoor', 'archaelogical_excavation': 'outdoor', 'archive': 'indoor', 'arena/hockey': 'indoor', 'arena/performance': 'indoor', 'arena/rodeo': 'outdoor', 'army_base': 'outdoor', 'art_gallery': 'indoor', 'art_school': 'indoor', 'art_studio': 'indoor', 'artists_loft': 'outdoor', 'assembly_line': 'indoor', 'athletic_field/outdoor': 'outdoor', 'atrium/public': 'outdoor', 'attic': 'indoor', 'auditorium': 'indoor', 'auto_factory': 'indoor', 'auto_showroom': 'indoor', 'badlands': 'outdoor', 'bakery/shop': 'indoor', 'balcony/exterior': 'outdoor', 'balcony/interior': 'outdoor', 'ball_pit': 'indoor', 'ballroom': 'outdoor', 'bamboo_forest': 'outdoor', 'bank_vault': 'indoor', 'banquet_hall': 'indoor', 'bar': 'indoor', 'barn': 'outdoor', 'barndoor': 'outdoor', 'baseball_field': 'outdoor', 'basement': 'indoor', 'basketball_court/indoor': 'indoor', 'bathroom': 'indoor', 'bazaar/indoor': 'indoor', 'bazaar/outdoor': 'outdoor', 'beach': 'outdoor', 'beach_house': 'outdoor', 'beauty_salon': 'indoor', 'bedchamber': 'indoor', 'bedroom': 'indoor', 'beer_garden': 'outdoor', 'beer_hall': 'indoor', 'berth': 'indoor', 'biology_laboratory': 'indoor', 'boardwalk': 'outdoor', 'boat_deck': 'outdoor', 'boathouse': 'outdoor', 'bookstore': 'indoor', 'booth/indoor': 'indoor', 'botanical_garden': 'outdoor', 'bow_window/indoor': 'indoor', 'bowling_alley': 'indoor', 'boxing_ring': 'indoor', 'bridge': 'outdoor', 'building_facade': 'outdoor', 'bullring': 'outdoor', 'burial_chamber': 'outdoor', 'bus_interior': 'indoor', 'bus_station/indoor': 'outdoor', 'butchers_shop': 'indoor', 'butte': 'outdoor', 'cabin/outdoor': 'outdoor', 'cafeteria': 'indoor', 'campsite': 'outdoor', 'campus': 'outdoor', 'canal/natural': 'outdoor', 'canal/urban': 'outdoor', 'candy_store': 'indoor', 'canyon': 'outdoor', 'car_interior': 'indoor', 'carrousel': 'outdoor', 'castle': 'outdoor', 'catacomb': 'indoor', 'cemetery': 'outdoor', 'chalet': 'outdoor', 'chemistry_lab': 'indoor', 'childs_room': 'indoor', 'church/indoor': 'indoor', 'church/outdoor': 'outdoor', 'classroom': 'indoor', 'clean_room': 'indoor', 'cliff': 'outdoor', 'closet': 'indoor', 'clothing_store': 'indoor', 'coast': 'outdoor', 'cockpit': 'indoor', 'coffee_shop': 'indoor', 'computer_room': 'indoor', 'conference_center': 'indoor', 'conference_room': 'indoor', 'construction_site': 'outdoor', 'corn_field': 'outdoor', 'corral': 'outdoor', 'corridor': 'indoor', 'cottage': 'outdoor', 'courthouse': 'outdoor', 'courtyard': 'outdoor', 'creek': 'outdoor', 'crevasse': 'outdoor', 'crosswalk': 'outdoor', 'dam': 'outdoor', 'delicatessen': 'indoor', 'department_store': 'indoor', 'desert/sand': 'outdoor', 'desert/vegetation': 'outdoor', 'desert_road': 'outdoor', 'diner/outdoor': 'outdoor', 'dining_hall': 'indoor', 'dining_room': 'indoor', 'discotheque': 'indoor', 'doorway/outdoor': 'outdoor', 'dorm_room': 'indoor', 'downtown': 'outdoor', 'dressing_room': 'indoor', 'driveway': 'outdoor', 'drugstore': 'indoor', 'elevator/door': 'indoor', 'elevator_lobby': 'indoor', 'elevator_shaft': 'indoor', 'embassy': 'outdoor', 'engine_room': 'indoor', 'entrance_hall': 'outdoor', 'escalator/indoor': 'indoor', 'excavation': 'outdoor', 'fabric_store': 'indoor', 'farm': 'outdoor', 'fastfood_restaurant': 'indoor', 'field/cultivated': 'outdoor', 'field/wild': 'outdoor', 'field_road': 'outdoor', 'fire_escape': 'outdoor', 'fire_station': 'outdoor', 'fishpond': 'indoor', 'flea_market/indoor': 'outdoor', 'florist_shop/indoor': 'indoor', 'food_court': 'indoor', 'football_field': 'outdoor', 'forest/broadleaf': 'outdoor', 'forest_path': 'outdoor', 'forest_road': 'outdoor', 'formal_garden': 'outdoor', 'fountain': 'outdoor', 'galley': 'indoor', 'garage/indoor': 'indoor', 'garage/outdoor': 'outdoor', 'gas_station': 'outdoor', 'gazebo/exterior': 'outdoor', 'general_store/indoor': 'indoor', 'general_store/outdoor': 'outdoor', 'gift_shop': 'indoor', 'glacier': 'outdoor', 'golf_course': 'outdoor', 'greenhouse/indoor': 'indoor', 'greenhouse/outdoor': 'outdoor', 'grotto': 'indoor', 'gymnasium/indoor': 'indoor', 'hangar/indoor': 'indoor', 'hangar/outdoor': 'outdoor', 'harbor': 'outdoor', 'hardware_store': 'indoor', 'hayfield': 'outdoor', 'heliport': 'outdoor', 'highway': 'outdoor', 'home_office': 'indoor', 'home_theater': 'indoor', 'hospital': 'indoor', 'hospital_room': 'indoor', 'hot_spring': 'outdoor', 'hotel/outdoor': 'outdoor', 'hotel_room': 'indoor', 'house': 'indoor', 'hunting_lodge/outdoor': 'outdoor', 'ice_cream_parlor': 'indoor', 'ice_floe': 'outdoor', 'ice_shelf': 'outdoor', 'ice_skating_rink/indoor': 'indoor', 'ice_skating_rink/outdoor': 'outdoor', 'iceberg': 'outdoor', 'igloo': 'outdoor', 'industrial_area': 'outdoor', 'inn/outdoor': 'outdoor', 'islet': 'outdoor', 'jacuzzi/indoor': 'indoor', 'jail_cell': 'indoor', 'japanese_garden': 'outdoor', 'jewelry_shop': 'indoor', 'junkyard': 'outdoor', 'kasbah': 'outdoor', 'kennel/outdoor': 'outdoor', 'kindergarden_classroom': 'indoor', 'kitchen': 'indoor', 'lagoon': 'outdoor', 'lake/natural': 'outdoor', 'landfill': 'outdoor', 'landing_deck': 'outdoor', 'laundromat': 'indoor', 'lawn': 'outdoor', 'lecture_room': 'indoor', 'legislative_chamber': 'indoor', 'library/indoor': 'indoor', 'library/outdoor': 'outdoor', 'lighthouse': 'outdoor', 'living_room': 'indoor', 'loading_dock': 'outdoor', 'lobby': 'indoor', 'lock_chamber': 'outdoor', 'locker_room': 'indoor', 'mansion': 'outdoor', 'manufactured_home': 'outdoor', 'market/indoor': 'indoor', 'market/outdoor': 'outdoor', 'marsh': 'outdoor', 'martial_arts_gym': 'indoor', 'mausoleum': 'outdoor', 'medina': 'outdoor', 'mezzanine': 'outdoor', 'moat/water': 'outdoor', 'mosque/outdoor': 'outdoor', 'motel': 'outdoor', 'mountain': 'outdoor', 'mountain_path': 'outdoor', 'mountain_snowy': 'outdoor', 'movie_theater/indoor': 'indoor', 'museum/indoor': 'indoor', 'museum/outdoor': 'outdoor', 'music_studio': 'indoor', 'natural_history_museum': 'indoor', 'nursery': 'indoor', 'nursing_home': 'indoor', 'oast_house': 'outdoor', 'ocean': 'outdoor', 'office': 'indoor', 'office_building': 'outdoor', 'office_cubicles': 'indoor', 'oilrig': 'outdoor', 'operating_room': 'indoor', 'orchard': 'outdoor', 'orchestra_pit': 'indoor', 'pagoda': 'outdoor', 'palace': 'outdoor', 'pantry': 'indoor', 'park': 'outdoor', 'parking_garage/indoor': 'indoor', 'parking_garage/outdoor': 'outdoor', 'parking_lot': 'outdoor', 'pasture': 'outdoor', 'patio': 'outdoor', 'pavilion': 'outdoor', 'pet_shop': 'indoor', 'pharmacy': 'indoor', 'phone_booth': 'outdoor', 'physics_laboratory': 'indoor', 'picnic_area': 'outdoor', 'pier': 'outdoor', 'pizzeria': 'indoor', 'playground': 'outdoor', 'playroom': 'indoor', 'plaza': 'outdoor', 'pond': 'outdoor', 'porch': 'indoor', 'promenade': 'outdoor', 'pub/indoor': 'indoor', 'racecourse': 'outdoor', 'raceway': 'outdoor', 'raft': 'outdoor', 'railroad_track': 'outdoor', 'rainforest': 'outdoor', 'reception': 'indoor', 'recreation_room': 'indoor', 'repair_shop': 'indoor', 'residential_neighborhood': 'outdoor', 'restaurant': 'indoor', 'restaurant_kitchen': 'indoor', 'restaurant_patio': 'indoor', 'rice_paddy': 'outdoor', 'river': 'outdoor', 'rock_arch': 'outdoor', 'roof_garden': 'outdoor', 'rope_bridge': 'outdoor', 'ruin': 'outdoor', 'runway': 'outdoor', 'sandbox': 'outdoor', 'sauna': 'indoor', 'schoolhouse': 'outdoor', 'science_museum': 'indoor', 'server_room': 'indoor', 'shed': 'outdoor', 'shoe_shop': 'indoor', 'shopfront': 'outdoor', 'shopping_mall/indoor': 'indoor', 'shower': 'indoor', 'ski_resort': 'outdoor', 'ski_slope': 'outdoor', 'sky': 'outdoor', 'skyscraper': 'outdoor', 'slum': 'outdoor', 'snowfield': 'outdoor', 'soccer_field': 'outdoor', 'stable': 'outdoor', 'stadium/baseball': 'indoor', 'stadium/football': 'indoor', 'stadium/soccer': 'indoor', 'stage/indoor': 'indoor', 'stage/outdoor': 'outdoor', 'staircase': 'indoor', 'storage_room': 'indoor', 'street': 'outdoor', 'subway_station/platform': 'indoor', 'supermarket': 'indoor', 'sushi_bar': 'indoor', 'swamp': 'outdoor', 'swimming_hole': 'outdoor', 'swimming_pool/indoor': 'indoor', 'swimming_pool/outdoor': 'outdoor', 'synagogue/outdoor': 'outdoor', 'television_room': 'outdoor', 'television_studio': 'outdoor', 'temple/asia': 'outdoor', 'throne_room': 'outdoor', 'ticket_booth': 'outdoor', 'topiary_garden': 'outdoor', 'tower': 'outdoor', 'toyshop': 'indoor', 'train_interior': 'indoor', 'train_station/platform': 'indoor', 'tree_farm': 'outdoor', 'tree_house': 'outdoor', 'trench': 'outdoor', 'tundra': 'outdoor', 'underwater/ocean_deep': 'outdoor', 'utility_room': 'indoor', 'valley': 'outdoor', 'vegetable_garden': 'outdoor', 'veterinarians_office': 'indoor', 'viaduct': 'outdoor', 'village': 'outdoor', 'vineyard': 'outdoor', 'volcano': 'outdoor', 'volleyball_court/outdoor': 'outdoor', 'waiting_room': 'indoor', 'water_park': 'outdoor', 'water_tower': 'outdoor', 'waterfall': 'outdoor', 'watering_hole': 'outdoor', 'wave': 'outdoor', 'wet_bar': 'indoor', 'wheat_field': 'outdoor', 'wind_farm': 'outdoor', 'windmill': 'outdoor', 'yard': 'outdoor', 'youth_hostel': 'indoor', 'zen_garden': 'outdoor'}


# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# images = glob.glob('atlanta/*.jpg')
# print(images)
db_table = 'scene_parsing_atlanta'
# db_table = 'scene_parsing_nyc'
images = glob.glob(r'/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/atlanta/*.jpg')
# images = glob.glob(r'/Users/divyachandana/Documents/NJIT/work/summertasks/jun1-jun5/nyc/**/*.jpg')
columns = ['imagepath','top1','top1_score','top2','top2_score','top3','top3_score','top4','top4_score','top5','top5_score','status']
columns_str = ','.join(columns)
i = 0
for img_name in images:
    try:
        i = i+1
    # load the test image
        filename = os.path.basename(img_name)
        file_check_query = "select count(*) from {} where imagepath like '%{}%'".format(db_table, filename)
        count = dbms.get_count_result(file_check_query)
        if count > 0: continue
        print(i)
        database_format = []
        database_format.append("'"+filename+"'")
        img = Image.open(img_name)
        input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

    # print('{} prediction on {}'.format(arch,img_name))
    # output the prediction
        for i in range(0, 5):
        # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
            database_format.append("'"+classes[idx[i]]+"'")
            database_format.append(format(probs[i],'.5f'))

        class_name = classes[idx[0]]
        status = places_365_dict[class_name]
        database_format.append("'"+status+"'")

        insert_query = "INSERT INTO {} ({}) VALUES ({});".format(db_table,columns_str,','.join(database_format))
        dbms.execute_query(insert_query)
    except:
        print("Error in parsing ", img_name)
