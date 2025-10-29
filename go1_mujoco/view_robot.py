import mujoco as mu 
from mujoco import viewer

def viewer():
    xml_path = "/home/siddarth/manipulation_ws/src/go1_mujoco/unitree_go2/scene_mjx.xml"
    model = mu.MjModel.from_xml_path(xml_path)
    data = mu.MjData(model)
    with mu.viewer.launch_passive(model, data) as sim_viewer:
        while sim_viewer.is_running():
            mu.mj_step(model, data)  # physics step
            site_id = model.site("RL_foot").id
            pos = data.site_xpos[site_id]
            rot = data.site_xmat[site_id].reshape(3,3)
            
            print("Foot position:", pos)
            print("Foot rotation matrix:\n", rot)
            sim_viewer.sync()  
            

if __name__ == "__main__":
    viewer()