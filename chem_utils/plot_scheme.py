import numpy as np
import networkx as nx
from scipy.optimize import minimize

import pandas as pd

import matplotlib.pyplot as plt
import PIL
from PIL import Image
import itertools

from yattag import Doc


def curve(x): return 0.5*(3*x-x**3)


def arrow(x_from, x_to, y_from, y_to, invert=False):
    if invert:
        y = np.linspace(-1, 1)
        x = curve(y)
    else:
        x = np.linspace(-1, 1)
        y = curve(x)
    return 0.5*(x_to-x_from)*(x+1)+x_from, 0.5*(y_to-y_from)*(y+1)+y_from


def html_curve(tag, x_from, x_to, y_from, y_to, x0, y0, name_from, name_to):
    dx = x_to-x_from
    dy = y_to-y_from
    dx = dx if abs(dx) > 0.03 else 0.03
    dy = dy if abs(dy) > 0.03 else 0.03
    if dy > 1:
        dy -= 1
        if dx < 0:
            dx = abs(dx)
            with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                     style=f"width:{int(120*dx)}px;height:{int(120*abs(dy))}px;position: absolute;top:{int(120*(y0-y_to+1))}px;left: {int(120*(x_to-x0)+50)}px;"):
                with tag('path', d=f"M 0 0 C 0 {int(0.5*100*dy)}, {int(120*dx)} {int(0.5*120*dy)}, {int(120*dx)} {int(120*abs(dy))}", stroke='red', fill="transparent"):
                    pass
        else:
            with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                     style=f"width:{int(120*dx)}px;height:{int(120*abs(dy))}px;position: absolute;top:{int(120*(y0-y_to+1))}px;left: {int(120*(x_from-x0)+50)}px;"):
                with tag('path', d=f"M 0 {int(120*abs(dy))} C 0 {int(0.5*100*dy)}, {int(120*dx)} {int(0.5*120*dy)}, {int(120*dx)} 0", stroke='red', fill="transparent"):
                    pass
    elif dy < -1:
        dy = abs(dy)
        dy -= 1
        if dx < 0:
            dx = abs(dx)
            with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                     style=f"width:{int(120*dx)}px;height:{int(120*dy)}px;position: absolute;top:{int(120*(y0-y_from+1))}px;left: {int(120*(x_to-x0)+50)}px;"):
                with tag('path', d=f"M {int(120*dx)} 0 C {int(120*dx)} {int(0.5*120*dy)}, 0 {int(0.5*120*dy)}, 0 {int(120*abs(dy))}", stroke='blue', fill="transparent"):
                    pass
        else:
            with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                     style=f"width:{int(120*dx)}px;height:{int(120*dy)}px;position: absolute;top:{int(120*(y0-y_from+1))}px;left: {int(120*(x_from-x0)+50)}px;"):
                with tag('path', d=f"M 0 0 C 0 {int(0.5*120*dy)}, {int(120*dx)} {int(0.5*120*dy)}, {int(120*dx)} {int(120*abs(dy))}", stroke='blue', fill="transparent"):
                    pass
    else:
        if dx < -0.83:
            dx = abs(dx)
            dx -= 0.83
            if dy > 0:
                with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                         style=f"width:{int(120*dx)}px;height:{int(120*dy)}px;position: absolute;top:{int(120*(y0-y_to)+60)}px;left: {int(120*(x_to-x0)+100)}px;"):
                    with tag('path', d=f"M {int(120*dx)} {int(120*abs(dy))} C {int(0.5*120*dx)} {int(120*abs(dy))}, {int(0.5*120*dx)} 0, 0 0", stroke='black', fill="transparent"):
                        pass
            else:
                dy = abs(dy)
                with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                         style=f"width:{int(120*dx)}px;height:{int(120*dy)}px;position: absolute;top:{int(120*(y0-y_from)+60)}px;left: {int(120*(x_to-x0)+100)}px;"):
                    with tag('path', d=f"M {int(120*dx)} 0 C {int(0.5*120*dx)} 0, {int(0.5*120*dx)} {int(120*dy)}, 0 {int(120*abs(dy))}", stroke='black', fill="transparent"):
                        pass
        elif dx > 0.83:
            dx -= 0.83
            if dy > 0:
                with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                         style=f"width:{int(120*dx)}px;height:{int(120*dy)}px;position: absolute;top:{int(120*(y0-y_to)+60)}px;left: {int(120*(x_from-x0)+100)}px;"):
                    with tag('path', d=f"M {int(120*dx)} 0 C {int(0.5*120*dx)} 0, {int(0.5*120*dx)} {int(120*dy)}, 0 {int(120*abs(dy))}", stroke='black', fill="transparent"):
                        pass
            else:
                dy = abs(dy)
                with tag('svg', id=f'{int(name_from)}_{int(name_to)}',
                         style=f"width:{int(120*dx)}px;height:{int(120*dy)}px;position: absolute;top:{int(120*(y0-y_from)+60)}px;left: {int(120*(x_from-x0)+100)}px;"):
                    with tag('path', d=f"M 0 0 C {int(0.5*120*dx)} 0, {int(0.5*120*dx)} {int(120*dy)}, {int(120*dx)} {int(120*abs(dy))}", stroke='black', fill="transparent"):
                        pass

        else:
            print(f'Warninig {int(name_from)}_{int(name_to)}')


def plot_scheme_html(df, ratio, save=None, show=True, show_x_axis=False):
    doc, tag, text = Doc().tagtext()
    page_width_px = int(120*(df.position_x.max()-df.position_x.min()))+100+120
    page_height_px = int(120*(df.position_y.max()-df.position_y.min()))+120+120

    with tag('html'):
        with tag("style"):
            text(f'''
            @page  {{
            margin: 0;
            size: {page_width_px}px {page_height_px}px
            }}
            ''')
        with tag('body', style=f"width:{page_width_px}px;height:{page_height_px}px;"):  # border:solid 4px black;
            # with tag('canvas', id="CanvasOfGeeks", style="border:solid 2px red;width:{}px;height:{}px;".format(int(100*(df.position_x.max()-df.position_x.min())), int(100*df.position_y.max()-df.position_y.min()))):
            #     pass
            x0 = df.position_x.min()-0.5
            y0 = df.position_y.max()+0.5
            with tag('div', id='diagram', style=f"width:{page_width_px}px;height:{page_height_px}px;"):
                for index, row in df.iterrows():
                    with tag('img', id=index, src='pics/{}.png'.format(index), alt=str(index),
                             style=f"width:100px;height:120px;position: absolute;top: {int(120*(y0-row.position_y))}px;left: {int(120*(row.position_x-x0))}px;"):
                        pass
                for index, row in df.iterrows():
                    if row.group != row.id:
                        x_to = row.position_x
                        y_to = row.position_y
                        x_from = df.loc[int(row.group)].position_x
                        y_from = df.loc[int(row.group)].position_y

                        html_curve(tag, x_from, x_to, y_from,
                                   y_to, x0, y0, row.group, row.id)

    result = doc.getvalue()

    if save is not None:
        file = open(save, "w")
        file.write(result)
        file.close()

    return result


def plot_scheme(df, ratio, save=None, show=True, show_x_axis=False):
    fig, ax = plt.subplots()
    for index, row in df.iterrows():
        # print(index, row.position_x,row.position_y)
        img = Image.open('{}/scheme_with_rate.png'.format(index))
        ax.imshow(img, extent=[row.position_x-0.5*ratio, row.position_x+0.5*ratio,
                  row.position_y-0.5, row.position_y+0.5], aspect=1)  # , interpolation='sinc'
    # ax.set_axis_off()

    for index, row in df.iterrows():
        # print(index)
        if row.group != row.id:
            x_from = row.position_x
            y_from = row.position_y
            x_to = df.loc[int(row.group)].position_x
            y_to = df.loc[int(row.group)].position_y
            dx = x_to-x_from
            dy = y_to-y_from
            linewidth = 0.5
            if dy > 1.:
                arr_x, arr_y = arrow(
                    x_from, x_to, y_from+0.5, y_to-0.5, invert=True)
                plt.plot(arr_x, arr_y, color='b', linewidth=linewidth)
            elif dy < -1.:
                arr_x, arr_y = arrow(
                    x_from, x_to, y_from-0.5, y_to+0.5, invert=True)
                plt.plot(arr_x, arr_y, color='r', linewidth=linewidth)
            elif dx > 0:
                arr_x, arr_y = arrow(
                    x_from+ratio/2, x_to-ratio/2, y_from, y_to)
                plt.plot(arr_x, arr_y, color='k', linewidth=linewidth)
            else:
                arr_x, arr_y = arrow(
                    x_from-ratio/2, x_to+ratio/2, y_from, y_to)
                plt.plot(arr_x, arr_y, color='k', linewidth=linewidth)

            # color= 'r' if dy>0 else 'b'

            # angle = np.arctan2(dy, dx)
            # if angle > -3*np.pi/4 and angle <= -np.pi/4:
            #     y_from -= 1/2
            #     y_to += 1/2
            # elif angle > -np.pi/4 and angle <= np.pi/4:
            #     x_from += ratio/2
            #     x_to -= ratio/2
            # elif angle > np.pi/4 and angle <= 3*np.pi/4:
            #     y_from += 1/2
            #     y_to -= 1/2
            # else:
            #     x_from -= ratio/2
            #     x_to += ratio/2

            # plt.arrow(x_from,y_from,x_to-x_from,y_to-y_from)

    plt.tight_layout()
    plt.xlim([df.position_x.min()-0.5*ratio, df.position_x.max()+0.5*ratio])
    plt.ylim([df.position_y.min()-0.5, df.position_y.max()+0.5])
    plt.grid(axis='y', zorder=0, linewidth=0.5,
             linestyle='--', color='xkcd:light grey')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if not show_x_axis:
        ax.set_xticklabels([])
        ax.set_xticks([])

    fig.set_size_inches(9*(df.position_x.max() - df.position_x.min()+ratio),
                        9*(df.position_y.max()-df.position_y.min()+1))

    # fig.patch.set_facecolor('xkcd:light grey')
    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=100)
    if show:
        plt.show()
    return fig, ax


def elliptic_distance(a_x, a_y, b_x, b_y, w, h):
    x = np.abs(a_x-b_x)
    y = np.abs(a_y-b_y)
    return np.sqrt((x/w)**2+(y/h)**2)  # if x > w or y > h else np.nan


def repulsion(d, minimum=2., cut=0, force=20):
    if d > minimum:
        return 0.
    elif d > cut:
        return force*(1./(d-cut)-1./(minimum-cut))
    else:
        return np.nan


def attraction(d, minimum=2., force=0.3):
    return 0. if d < minimum else force*(d-minimum)


def energy_x_with_attraction(a, b, ratio, scaling=1.3):
    x = np.abs(a[0]-b[0])
    y = np.abs(a[1]-b[1])
    if y > scaling:
        return x
    d = x*scaling
    if d > 2.:
        return 0.5*d
    elif d > 1:
        return 1./(d-1.)
    else:
        return np.nan


def energy_x_without_attraction(a, b, ratio, scaling=1.3):
    x = np.abs(a[0]-b[0])
    y = np.abs(a[1]-b[1])
    if y > scaling:
        # print('y > scaling')
        # print(f'{x = }')
        return 1.
    d = x*scaling
    if d > 2.:
        return 1.
    elif d > 1:
        return 1./(d-1.)
    else:
        return np.nan


def energy_elliptic_with_attraction(a_x, a_y, b_x, b_y, ratio, minimum=1.):
    d = elliptic_distance(a_x, a_y, b_x, b_y, ratio, 1)
    return attraction(d, minimum=minimum)+repulsion(d, minimum=minimum)


test_energy_elliptic_with_attraction = np.vectorize(
    energy_elliptic_with_attraction, excluded=['a_x,a_y,ratio,minimum'])


def energy_elliptic_without_attraction(a_x, a_y, b_x, b_y, ratio, minimum=1.2):
    d = elliptic_distance(a_x, a_y, b_x, b_y, ratio, 1)
    return repulsion(d, minimum=minimum)


test_energy_elliptic_without_attraction = np.vectorize(
    energy_elliptic_without_attraction, excluded=['a_x,a_y,ratio,minimum'])


def energy_combined_with_attraction(a_x, a_y, b_x, b_y, ratio, minimum=1.2, force=1.0):
    x = np.abs(a_x-b_x)
    y = np.abs(a_y-b_y)
    # if y > minimum:
    #     return x
    # else:
    d = elliptic_distance(a_x, a_y, b_x, b_y, ratio, 1)
    return repulsion(d, minimum=minimum)+force*x


test_energy_combined_with_attraction = np.vectorize(
    energy_combined_with_attraction, excluded=['a_x,a_y,ratio,minimum, force'])


def group_enegry(df, ratio, group, x=None):
    _df = df.copy()
    if x is not None:
        # print(f'{x = }')
        _df.loc[_df.group == group, 'position_x'] = x
        # pos = x.reshape(-1, 2)
        # _df.loc[_df.group == group, 'position_x'] = pos[:, 0]
        # _df.loc[_df.group == group, 'position_y'] = pos[:, 1]
    energy = 0.
    for (i, row_i), (j, row_j) in itertools.combinations(_df.loc[_df.group == group].iterrows(), 2):
        if i == group or j == group:
            # print(f'{i = }')
            # print(f'{j = }')
            energy += energy_combined_with_attraction(
                row_i.position_x, row_i.position_y, row_j.position_x, row_j.position_y, ratio, minimum=1.7, force=4.0)
        else:
            # print(f'{i = }')
            # print(f'{j = }')
            energy += energy_elliptic_without_attraction(
                row_i.position_x, row_i.position_y, row_j.position_x, row_j.position_y, ratio, minimum=1.5)

    # for index, row in _df.iterrows():
    #     energy += 10*np.abs(row.log_t-row.position_y)

    # energy += 0.01*(_df.position_x.max()-_df.position_x.min()) * \
    #     (_df.position_y.max()-_df.position_y.min())/len(set(_df.group.values))

    # print(f'{energy = }')

    return energy


def energy_intergroup_distance(df, ratio, x=None):
    if x is not None:
        rx = list(itertools.accumulate(x))
        # print(list(rx))
        # print(list(set(df.group.values)))
        # print(len(list(rx)), len(list(set(df.group.values))))
        # print(list(zip(list(rx), list(set(df.group.values)))))
        for offset, group in zip(list(rx), list(set(df.group.values))):
            # print(f'{offset = } {group = }')
            df.loc[df.group == group, 'position_x'] += offset

    energy = 0.
    for (i, row_i), (j, row_j) in itertools.combinations(df.iterrows(), 2):
        if row_i.group != row_j.group:
            if row_i.group == i and row_j.group == j:
                energy += energy_elliptic_without_attraction(
                    row_i.position_x, row_i.position_y, row_j.position_x, row_j.position_y, ratio, minimum=2.5)
            else:
                energy += energy_elliptic_without_attraction(
                    row_i.position_x, row_i.position_y, row_j.position_x, row_j.position_y, ratio, minimum=1.5)
    # print(f'{energy}')
    # print(f'{df.position_x.max() = } {df.position_x.min() = }')
    # print(100*((df.position_x.max()-df.position_x.min())/(len(set(df.group.values))*3))**4)
    energy += 100*((df.position_x.max()-df.position_x.min()) /
                   (len(set(df.group.values))*3))**4

    if x is not None:
        for offset, group in zip(list(rx), list(set(df.group.values))):
            df.loc[df.group == group, 'position_x'] -= offset
    # print(f'{energy = }')
    return energy


def full_energy(df, ratio):
    energy = 0.0
    for group in set(df.group.values):
        energy += group_enegry(df, ratio, group)
    energy += energy_intergroup_distance(df, ratio)
    return energy


def initial_guess(df, ratio):
    df['position_x'] = -ratio*1.5
    df['position_y'] = df.log_t.values
    for group in list(set(df.group.values)):
        index_list = [index for index,
                      row in df.iterrows() if row.group == int(group)]
        if int(group) in index_list:
            index_list.remove(int(group))
            index_list.insert(int(len(index_list)/2), int(group))
        for index in index_list:
            df.loc[index, 'position_x'] = df['position_x'].max()+ratio*1.5

        # df.loc[df.group == group, 'position_x'] = 2 * \
        #     (np.random.rand(len(df.loc[df.group == group]))-0.5)+i*5


def initial_guess_old(df):
    groups = [df.id[df.group == i] for i in set(df.group.values)]
    G = nx.Graph()
    for group in groups:
        H = nx.complete_graph(group)
        # print(H.nodes)
        G = nx.disjoint_union(G, H)
    # print(G)
    pos = nx.spring_layout(G, k=1.2, seed=np.random.seed(seed=0), dim=2)
    # print(pos)
    positions = np.array(list(pos.values()))*5
    return np.round(positions[:, 0], 2), df.log_t.values
    # return np.round(positions[:, 0], 2), np.round(positions[:, 1], 2)


def minimize_groups(df, ratio, method='BFGS'):
    for group in set(df.group.values):
        # print(f'{group = }')
        # print(df.group == group)
        # print(f'{group_enegry(df, ratio, group) = }')
        # positions = np.array([df.loc[df.group == group].position_x.values,
        #                      df.loc[df.group == group].position_y.values]).T
        positions = df.loc[df.group == group].position_x.values
        positions = positions.reshape(-1)
        res = minimize(lambda x: group_enegry(df, ratio, group, x),
                       positions, tol=0.01, method=method)
        # print(f'{group_enegry(df, ratio, group, res.x) = }')
        # pos = res.x.reshape(-1, 2)
        df.loc[df.group == group, 'position_x'] = res.x
        # df.loc[df.group == group, 'position_x'] = pos[:, 0]
        # df.loc[df.group == group, 'position_y'] = pos[:, 1]


def minimize_intergroup_distance(df, ratio, method='BFGS'):
    res = minimize(lambda x: energy_intergroup_distance(df, ratio, x),
                   np.zeros(len(set(df.group.values))), tol=0.01, method=method)
    print(res)
    rx = list(itertools.accumulate(res.x))
    for offset, group in zip(rx, set(df.group.values)):
        print(f'{offset = } {group = }')
        df.loc[df.group == group, 'position_x'] += offset

def round_positions(df):
    df.position_x=np.round(df.position_x,2)
    df.position_y=np.round(df.position_y,2)

def minimize_scheme(df, ratio, save='fig.pdf', method='BFGS'):
    initial_guess(df, ratio)
    minimize_groups(df, ratio, method=method)
    minimize_intergroup_distance(df, ratio, method=method)
    round_positions(df)
