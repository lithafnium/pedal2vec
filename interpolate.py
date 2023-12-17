import os
import json

folder_root = "./SmartPedal_Tones"
out_folder = "./interpolated_tones"

INTERPOLATE_FACTOR = 4

if __name__ == "__main__":
    pedals = os.listdir(folder_root)

    for p1 in range(len(pedals)):
        for p2 in range(p1 + 1, len(pedals)):
            p_1 = os.path.join(folder_root, pedals[p1])
            p_2 = os.path.join(folder_root, pedals[p2])

            name = f'{pedals[p1].replace(".json", "")}_x_{pedals[p2].replace(".json", "")}'
            
            
            with open(p_1, "r") as pedal1: 
                pedal1_weights = json.load(pedal1)
            with open(p_2, "r") as pedal2: 
                pedal2_weights = json.load(pedal2)

            res1 = pedal1_weights["residual_channels"]
            res2 = pedal2_weights["residual_channels"]

            if res1 != res2: 
                continue

            dilations1 = pedal1_weights["dilations"]
            dilations2 = pedal2_weights["dilations"]

            if sorted(dilations1) != sorted(dilations2):
                continue


            different = False

            print(name)
            # print("========== RESIDUAL CHANNELS")
            # print(pedal1_weights["residual_channels"], pedal2_weights["residual_channels"])
            
            # print("========== LAYERS")
            interpolated_plugins = {}
            for i in range(1, INTERPOLATE_FACTOR):
                interpolated_plugin = {k:v for k, v in pedal1_weights.items()}
                interpolated_variables = []

                # loop through all avariables
                for (v1, v2) in zip(pedal1_weights["variables"], pedal2_weights["variables"]):
                    if len(v1["data"]) != len(v2["data"]):
                        different = True
                        break

                    data1 = v1["data"]
                    data2 = v2["data"]
                    new_var = {k:v for k, v in v1.items()}
                    new_data = []

                    for (d1, d2) in zip(data1, data2):
                        d1 = float(d1)
                        d2 = float(d2) 

                        interpolate_number = abs(d2 - d1) / INTERPOLATE_FACTOR
                        if d1 < d2: 
                            new_val = d1 + interpolate_number * i
                        elif d1 > d2: 
                            new_val = d1 - interpolate_number * i
                        else: 
                            new_val = d1

                        new_data.append(str(new_val))
                    

                        # new_data.append(str((float(d1) + float(d2)) / 2))
                    new_var["data"] = new_data
                    interpolated_variables.append(new_var)

                interpolated_plugin["variables"] = interpolated_variables

                interpolated_plugins[i] = interpolated_plugin
                if different:
                    break
            
            if different:
                continue
            

            for interpolation_factor, data in interpolated_plugins.items():
                path = os.path.join(out_folder, name, str(interpolation_factor))
                if not os.path.exists(path):
                    os.makedirs(path)
                
                print(f"Writing {name} with interpolate factor {interpolation_factor}...")
                with open(os.path.join(path, f"{name}_{interpolation_factor}.json"), "w") as f:
                    json.dump(data, f, indent=4)
            
            input("continue...")


    # for pedal in pedals: 
    #     file = os.path.join(folder_root, pedal)
