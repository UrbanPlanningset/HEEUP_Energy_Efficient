from pyenergyplus.api import EnergyPlusAPI
a = "FullServiceRestaurant"
building_type = a
idf_file_name = f'RefBldg+{building_type}+New2004_v1.3_5.0_1A_USA_FL_MIAMI.idf'
api = EnergyPlusAPI()
state = api.state_manager.new_state()
a = api.runtime.run_energyplus(state, [
    '-d', r'Eneryplus_Datasets',
    '-w', '1A_USA_FL_MIAMI.epw',
    '-r', idf_file_name,
])
print(a)

