import pyvista as pv

pl = pv.Plotter()
pl.add_mesh(pv.Cone(center=(0, 0, 0)), name='Cone')
pl.add_mesh(pv.Cube(center=(1, 0, 0)), name='Cube')
pl.add_mesh(pv.Sphere(center=(1, 1, 0)), name='')
pl.add_mesh(pv.Cylinder(center=(0, 1, 0)), name='Cylinder')


def reset():
    for a in pl.renderer.actors.values():
        if isinstance(a, pv.Actor):
            a.prop.color = 'lightblue'
            a.prop.show_edges = False


def callback(actor):
    reset()
    actor.prop.color = 'green'
    actor.prop.show_edges = True
    print(f'{actor = }')


pl.enable_mesh_picking(callback, use_actor=True, show=False)
pl.show()
