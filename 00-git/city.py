from safety.safety import draw_safety
from leisure.leisure import draw_leisure
from outdoors.outdoors import draw_outdoors
from education.education import draw_education
from infrastructure.road import draw_road

def draw_city():
    draw_safety()
    draw_road()
    draw_leisure()
    draw_road()
    draw_outdoors()
    draw_road()
    draw_education()
    return

draw_city()
