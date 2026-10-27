import copy
import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[3]

def build_scene(config):
    source = ET.parse(ROOT / 'assets/franka_emika_panda/panda.xml').getroot()
    root = ET.Element('mujoco', model='dual_panda_stack')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep=str(config['physics_dt']), integrator='implicitfast', cone='elliptic')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='640', offheight='480')
    ET.SubElement(visual, 'headlight', ambient='0.4 0.4 0.4')
    root.append(copy.deepcopy(source.find('default')))
    assets = copy.deepcopy(source.find('asset'))
    for mesh in assets.findall('mesh'):
        mesh.set('name', mesh.get('name', pathlib.Path(mesh.get('file')).stem))
        mesh.set('file', str(ROOT / 'assets/franka_emika_panda/assets' / mesh.get('file')))
    root.append(assets)
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -1 3', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 .1', rgba='.18 .2 .23 1')
    top=config['table_top']
    ET.SubElement(world, 'geom', name='table', type='box', pos=f'.05 0 {top-.04}', size='.8 .85 .04', rgba='.65 .57 .45 1', friction='1 .005 .0001')
    ET.SubElement(world, 'camera', **({'name':'overview'} | config['cameras']['overview']))
    refs = {'name', 'joint', 'joint1', 'joint2', 'tendon', 'body1', 'body2'}
    for side, y in [('left', -.38), ('right', .38)]:
        body = copy.deepcopy(source.find('worldbody/body'))
        body.set('pos', f'-.45 {y} {top}')
        for element in body.iter():
            for attr in refs:
                if attr in element.attrib:
                    element.set(attr, side + '_' + element.get(attr))
            if element.tag == 'body':
                element.set('gravcomp', '1')
            if element.tag == 'geom' and element.get('class') != 'visual':
                element.set('friction', '1.5 .01 .001')
                element.set('condim', '4')
        hand = body.find(f'.//body[@name="{side}_hand"]')
        ET.SubElement(hand, 'site', name=side+'_ee', pos='0 0 .1034', size='.005', rgba='0 1 0 0')
        ET.SubElement(hand, 'camera', **({'name':side+'_wrist'} | config['cameras'][side+'_wrist']))
        world.append(body)
        for section in ['actuator', 'tendon', 'equality', 'contact']:
            dest = root.find(section)
            if dest is None:
                dest = ET.SubElement(root, section)
            for original in source.find(section):
                child = copy.deepcopy(original)
                for element in child.iter():
                    for attr in refs:
                        if attr in element.attrib:
                            element.set(attr, side + '_' + element.get(attr))
                dest.append(child)
    sa,sb=config['a_half_size'],config['b_half_size']
    friction=' '.join(map(str,config['contact_friction']))
    a = ET.SubElement(world, 'body', name='block_a', pos=f'0 -.38 {top+sa}')
    ET.SubElement(a, 'freejoint', name='a_free')
    ET.SubElement(a, 'geom', name='a_geom', type='box', size=f'{sa} {sa} {sa}', mass=str(config['a_mass']), rgba='.9 .12 .08 1', friction=friction, condim='4', solref='.008 1')
    b = ET.SubElement(world, 'body', name='block_b', pos=f'.2 -.3 {top+sb}')
    ET.SubElement(b, 'geom', name='b_geom', type='box', size=f'{sb} {sb} {sb}', rgba='.08 .25 .85 1', friction=friction, condim='4', solref='.008 1')
    return ET.tostring(root, encoding='unicode')
