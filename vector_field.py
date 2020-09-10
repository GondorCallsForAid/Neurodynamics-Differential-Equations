from manimlib.imports import *



class ThetaVsTAxes(Axes):
    CONFIG = {
        "x_min": 0,
        "x_max": 5, #8,
        "y_min": 0, #-PI / 2,
        "y_max": 7, #PI / 2,
        "y_axis_config": {
            "tick_frequency":  1,
            "unit_size": 1.5,
        },
        "axis_config": {
            "color": "#EEEEEE",
            "stroke_width": 2,
            "include_tip": False,
        },
        "graph_style": {
            "stroke_color": BLUE,
            "stroke_width": 3,
            "fill_opacity": 0,
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_labels()

    def add_axes(self):
        self.axes = Axes(**self.axes_config)
        #self.axes.shift(np.array([0,2,0]))
        self.add(self.axes)

    def add_labels(self):
        x_axis = self.get_x_axis()
        y_axis = self.get_y_axis()

        t_label = self.t_label = TexMobject("t")
        t_label.next_to(x_axis.get_right(), UP, MED_SMALL_BUFF)
        x_axis.label = t_label
        x_axis.add(t_label)
        theta_label = self.theta_label = TexMobject("y(t)")
        theta_label.next_to(y_axis.get_top(), UP, SMALL_BUFF)
        y_axis.label = theta_label
        y_axis.add(theta_label)

        self.y_axis_label = theta_label
        self.x_axis_label = t_label

        x_axis.add_numbers()
        y_axis.add(self.get_y_axis_coordinates(y_axis))

    def get_y_axis_coordinates(self, y_axis):
        texs = [
            # "\\pi \\over 4",
            # "\\pi \\over 2",
            # "3 \\pi \\over 4",
            # "\\pi",
            "",
            "",
            "",
            "",
            "",
            ""
        ]
        values = np.arange(1, 6) 
        labels = VGroup()
        for pos_tex, pos_value in zip(texs, values):
            neg_tex = "-" + pos_tex
            neg_value = -1 * pos_value
            for tex, value in (pos_tex, pos_value), (neg_tex, neg_value):
                if value > self.y_max or value < self.y_min:
                    continue
                symbol = TexMobject(tex)
                symbol.scale(0.5)
                point = y_axis.number_to_point(value)
                symbol.next_to(point, LEFT, MED_SMALL_BUFF)
                labels.add(symbol)
        return labels

    # live drawn graph of y(t) which represents the path of the dot
    def get_live_drawn_graph(self, dot,
                             t_max=None,
                             t_step=1.0 / 60,
                             **style):
        style = merge_dicts_recursively(self.graph_style, style)
        if t_max is None:
            t_max = self.x_max

        graph = VMobject()
        graph.set_style(**style)

        graph.all_coords = [(0, dot.get_center()[0])]
        graph.time = 0
        graph.time_of_last_addition = 0

        def update_graph(graph, dt):
            graph.time += dt
            if graph.time > t_max:
                graph.remove_updater(update_graph)
                return
            new_coords = (graph.time, dot.get_center()[0])
            if graph.time - graph.time_of_last_addition >= t_step:
                graph.all_coords.append(new_coords)
                graph.time_of_last_addition = graph.time
            points = [
                self.coords_to_point(*coords)
                for coords in [*graph.all_coords, new_coords]
            ]
            graph.set_points_smoothly(points)

        graph.add_updater(update_graph)
        return graph




class Grid(VGroup):
    CONFIG = {
        "height": 6.0,
        "width": 6.0,
    }

    def __init__(self, rows, columns, **kwargs):
        digest_config(self, kwargs, locals())
        super().__init__(**kwargs)

        x_step = self.width / self.columns
        y_step = self.height / self.rows

        for x in np.arange(0, self.width + x_step, x_step):
            self.add(Line(
                [x - self.width / 2., -self.height / 2., 0],
                [x - self.width / 2., self.height / 2., 0],
            ))
        for y in np.arange(0, self.height + y_step, y_step):
            self.add(Line(
                [-self.width / 2., y - self.height / 2., 0],
                [self.width / 2., y - self.height / 2., 0]
            ))


class ScreenGrid(VGroup):
    CONFIG = {
        "rows": 8,
        "columns": 14,
        "height": FRAME_Y_RADIUS * 2,
        "width": 14,
        "grid_stroke": 0.5,
        "grid_color": WHITE,
        "x_axis_color": WHITE,
        "y_axis_color": WHITE,
        "x_axis_stroke": 2,
        "y_axis_stroke": 2,
        "labels_scale": 0.5,
        "labels_buff": 0,
        "number_decimals": 2
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rows = self.rows
        columns = self.columns
        grid = Grid(width=self.width, height=self.height, rows=rows, columns=columns)
        grid.set_stroke(self.grid_color, self.grid_stroke)

        vector_ii = ORIGIN + np.array((- self.width / 2, - self.height / 2, 0))
        vector_si = ORIGIN + np.array((- self.width / 2, self.height / 2, 0))
        vector_sd = ORIGIN + np.array((self.width / 2, self.height / 2, 0))

        x_axis = Line(LEFT * self.width / 2, RIGHT * self.width / 2)
        y_axis = Line(DOWN * self.height / 2, UP * self.height / 2)
        
        
        x_axis.set_stroke(self.x_axis_color, self.x_axis_stroke)
        y_axis.set_stroke(self.y_axis_color, self.y_axis_stroke)
        
        self.x_axis = x_axis
        self.y_axis = y_axis

        divisions_x = self.width / columns
        divisions_y = self.height / rows

        directions_buff_x = [UP, DOWN]
        directions_buff_y = [RIGHT, LEFT]
        dd_buff = [directions_buff_x, directions_buff_y]
        vectors_init_x = [vector_ii, vector_si]
        vectors_init_y = [vector_si, vector_sd]
        vectors_init = [vectors_init_x, vectors_init_y]
        divisions = [divisions_x, divisions_y]
        orientations = [RIGHT, DOWN]
        labels = VGroup()
        set_changes = zip([columns, rows], divisions, orientations, [0, 1], vectors_init, dd_buff)
        for c_and_r, division, orientation, coord, vi_c, d_buff in set_changes:
            for i in range(1, c_and_r):
                for v_i, directions_buff in zip(vi_c, d_buff):
                    ubication = v_i + orientation * division * i
                    coord_point = round(ubication[coord], self.number_decimals)
                    label = TextMobject(str(coord_point)).scale(self.labels_scale)
                    label.next_to(ubication, directions_buff, buff=self.labels_buff)
                    labels.add(label)

        self.add(grid, x_axis, y_axis, labels)


def HSL(hue,saturation=1,lightness=0.5):
    return Color(hsl=(hue,saturation,lightness))

def double_linear(t):
    if t < 0.5:
        return linear(t*2)
    else:
        return linear(1-(t-0.5)*2)
    

        
class VectorField_one_d(Scene):
    
    CONFIG = {
        "axes_config": {
            "y_axis_config": {"unit_size": 0.75},
            "x_axis_config": {
                "unit_size": 0.5,
                "numbers_to_show": range(2, 20, 2),
                "number_scale_val": 0.5,
            },
            "x_max": 20,
            "axis_config": {
                "tip_length": 0.3,
                "stroke_width": 2,
            }
        }
    }
    
    
    def get_hsl_set_colors(self,saturation=1,lightness=0.5):
        return [*[HSL(i/360,saturation,lightness) for i in range(120)]]
    
    def highlight_object(self, thing, size = 1.5):
        self.play(ApplyMethod(thing.scale, size))
        self.play(ApplyMethod(thing.scale, 1/size))
        

    
    def construct(self):
        
        screen_grid = ScreenGrid()
        self.add(screen_grid)
        
            
        def one_d_vc(a):
            # changing a changes the position or origin of the direction vector
            # to only have vectors along the x-axis we set a[1] to 0
            a[1] = 0
            
            # set x
            x = -a[0]
        
            # set other dimensions to 0 because we want to simulate a one dimensional system
            y = 0    
            z = 0
            
            # test to scale x-axis
            # a[0] *= 1.5
    
            return np.array([x,y,z])
    
    
        def exponential_func(p):
            p[1] = 0
            x = p[0]
            
            x_change = x
            
            return x * RIGHT


    
        # one dimensional vector field y' = -y
        vector_field_norm = VectorField(one_d_vc, 
                                        # scale distance between vectors
                                        delta_x =  1, 
                                        delta_y =  1,
                                    
                                        # scale the interval range
                                        min_magnitude = 0,
                                        max_magnitude = 5,
                                        
                                        # scale vector size
                                        length_func = lambda norm: 0.8 * sigmoid(norm), # 0.8
                                        
                                        # adjust vectors
                                        vector_config = {# "stroke_width": 6,     
                                                         # "buff": MED_SMALL_BUFF, # 0.25
                                                         # "max_tip_length_to_length_ratio": 0.25,
                                                         "max_tip_length_to_length_ratio": 0.35,
                                                         # "max_stroke_width_to_length_ratio": 5,
                                                         "max_stroke_width_to_length_ratio": 8,
                                                         # "preserve_tip_size_when_scaling": True,
                                                         # "preserve_tip_size_when_scaling": False,
                                                        }
                                        )
        
        # vector field of exponential function
        vector_field_exp = VectorField(exponential_func, 
                                        # scale distance between vectors
                                        delta_x =  1, 
                                        delta_y =  1,
                                    
                                        # scale the interval range
                                        min_magnitude = 0,
                                        max_magnitude = 5,
                                        
                                        # scale vector size
                                        length_func = lambda norm: 0.8 * sigmoid(norm),
                                        
                                        # adjust vectors
                                        vector_config = {
                                                         "max_tip_length_to_length_ratio": 0.35,
                                                         "max_stroke_width_to_length_ratio": 8,
                                                        }
                                        )
        
        
        # create differential equations
        diff_equation_1 = TexMobject("\\dot y", #0
                                   "(t)",       #1
                                   "&=",        #2
                                   "f",         #3
                                   "(",         #4
                                   "y",         #5
                                   "(t))"       #6
                                  ).scale(2)
                
        diff_equation_1.move_to(np.array([0,2,0]))
        diff_equation_1[5].set_color(BLUE)
        
        
        diff_equation_2 = TexMobject("\\dot y", #0
                                     "(t)",     #1
                                     "&=",      #2
                                     "-",       #3
                                     "y",       #4
                                     "(t)"      #5
                                  ).scale(2)
        
        diff_equation_2.move_to(np.array([0,2,0]))
        diff_equation_2[4].set_color(BLUE)
        
        
        diff_equation_exp = TexMobject("\\dot y", #0
                                       "(t)",     #1
                                       "&=",      #2
                                       "y",       #3
                                       "(t)"      #4
                                  ).scale(2)
        
        diff_equation_exp.move_to(np.array([0,2,0]))
        diff_equation_exp[3].set_color(BLUE)
        
        diff_equation_exp_solution = TexMobject("\\dot {e^t}", #0
                                                "&=",        #1
                                                "e^t",       #2
                                  ).scale(2)
        
        diff_equation_exp_solution.move_to(np.array([0,2,0]))
        diff_equation_exp_solution[2].set_color(BLUE)
        diff_equation_exp_solution[0].set_color(self.get_hsl_set_colors())
        
       
        
        # display differntial equation 
        self.play(Write(diff_equation_1))
        self.wait(10)
        
        # shift y(t) to x-axis
        # highlight x-axis as space for y(t) values
        screen_grid.x_axis.generate_target()
        screen_grid.x_axis.target.set_stroke(BLUE, 6)
        y_t = diff_equation_2[4:6].copy()
        self.play(
                  ApplyMethod(y_t.shift, np.array([4.5,-1.5,0])),
                  MoveToTarget(screen_grid.x_axis)
                 )
        self.play(
                  ApplyMethod(y_t.scale, 0.5),
                 )
        self.wait(5)
        screen_grid.x_axis.target.set_stroke(BLUE, 2)
        self.play(MoveToTarget(screen_grid.x_axis))
        self.wait(4)
        
        
        # make f rainbow colors because its responsible for the arrows
        self.play(ApplyMethod(diff_equation_1[0].set_color, self.get_hsl_set_colors()))
        #diff_equation_1[0].set_color(color=self.get_hsl_set_colors())
        
        
        # display vector field (line) for y' = -y 
        #self.play(*[GrowArrow(vec) for vec in vector_field_norm])
        #self.wait(4)

        # reveal differential equation
        #self.play(Transform(diff_equation_1[3:7], diff_equation_2[3:6]))
        #self.wait(10)
        
        
        # display vector field for y' = y
        self.play(*[GrowArrow(vec) for vec in vector_field_exp])
        self.wait(19)
        
        # highlight function f
        self.highlight_object(diff_equation_1[3], size = 1.7)
        
        self.wait(49)
        
        
        # reveal differential equation
        self.play(Transform(diff_equation_1[3:7], diff_equation_exp[3:5]))
        self.wait(2)
        
        dot_at_zero = Dot([0,0,0], color=WHITE).scale(2)
        self.play(GrowFromPoint(dot_at_zero, dot_at_zero.get_center()))
        self.wait(5)
        self.play(FadeOut(dot_at_zero))
        self.wait(21)
        
        
        
        # show example of a function that does not satisfy the differential equation
        dot_simple_func = Dot([4,0,0], color=WHITE).scale(2)
        self.play(DrawBorderThenFill(dot_simple_func))
        self.wait(5)
        simple_function = TexMobject("0 \\; \\ ",
                                     "\\neq \\quad",
                                     "4"
                                    ).scale(2)
        
        simple_function.move_to(np.array([-0.3,-2,0]))
        self.play(Write(simple_function[2]))
        self.wait(5)
        self.play(Write(simple_function[0:2]))
        self.wait(10)
        
        self.play(FadeOut(simple_function),
                  FadeOut(dot_simple_func))
        self.wait(3)

        
        
        # move dot along vectors
        # dot1 = Dot([5.5,0,0], color=WHITE).scale(2)
        dot1 = Dot([0.01,0,0], color=WHITE).scale(2)
        self.play(DrawBorderThenFill(dot1))
        
        self.wait(10)
        
        
        
        
        
        
        
        axes = ThetaVsTAxes(**self.axes_config)
        axes.move_to(np.array([0.14,0.01,0]))
        self.axes = axes
        
        self.wait()
        graph = axes.get_live_drawn_graph(dot1)
        self.add(axes)
        self.add(graph)
        for dot in dot1: #,dot2:
            move_submobjects_along_vector_field(
                dot,
                lambda p: exponential_func(p)
            )
        self.wait(10)
        for dot in dot1: #,dot2:
            dot.clear_updaters()
        self.wait(4)
        
        
        
        self.play(Transform(diff_equation_1[0:6], diff_equation_exp_solution[0:3]))
        
        self.wait(7)
        
       
        # clear screen except grid
        self.play(FadeOut(diff_equation_1),
                  FadeOut(graph),
                  FadeOut(axes),
                  FadeOut(vector_field_exp),
                  FadeOut(y_t))
        self.wait(1)
        
        
        
        

def simple_two_d(p):
    x, y = p[:2]
    x_change = x
    y_change = y
    return x_change * RIGHT + y_change * UP


def functioncurlreal(p, velocity=0.05):
    x, y = p[:2]
    x_change = -y * 0.5
    y_change = x * 0.5
    return x_change * RIGHT + y_change * UP

def advanced_two_d(p):
    x, y = p[:2]
    x_change = (x + y)/2
    y_change = (x - y)/4
    return x_change * RIGHT + y_change * UP
        
        
        
class VectorField_two_d(Scene):
    def construct(self):
        
        # general case of two dimensional equation
        x_change = TexMobject("\\dot x", " = ", "f(x,y)").scale(1.6).move_to(np.array([0,2.2,0]))
        y_change = TexMobject("\\dot y", " = ", "g(x,y)").scale(1.6).move_to(np.array([0,0.7,0]))
        
        x_change[0].set_color(ORANGE)
        y_change[0].set_color(PURPLE)
        
        x_c_copy = TexMobject(
                              "\\dot x", 
                              " = ", 
                              "{x+y",    
                              "\\over",
                              "2}"
                             ).scale(1.6)
        y_c_copy = TexMobject(
                              "\\dot y", 
                              " = ", 
                              "{x-y",    
                              "\\over",
                              "4}"
                             ).scale(1.6)
        
        x_c_copy[0].set_color(ORANGE)
        y_c_copy[0].set_color(PURPLE)
        
        
        
        
        diff_advanced = TexMobject(
                               "V",        #0
                               "(x,y)",    #1
                               "&=",       #2
                               "\\langle", #3
                               "{x+y",     #4
                               "\\over",
                               "2}"
                               ",",        #5
                               "{x-y",     #6
                               "\\over",
                               "4}",
                               "\\rangle"  #7
                              )
        
        
        
        # different notation
        diff_diff = TexMobject(
                               "V",        #0
                               "(x,y)",    #1
                               "&=",       #2
                               "\\langle", #3
                               "f(x,y)",   #4
                               ",",        #5
                               "g(x,y)",   #6
                               "\\rangle"  #7
                              )
        diff_diff.scale(1.6).move_to(np.array([0,1.5,0]))
            
        
        # simple differential equation
        x_change_simple = TexMobject("\\dot x = x").scale(1.6).move_to(np.array([0,2.2,0]))
        y_change_simple = TexMobject("\\dot y = y").scale(1.6).move_to(np.array([0,0.7,0]))
        
        
        # simple differential equation with different notation
        diff_simple = TexMobject(
                               "V",        #0
                               "(x,y)",    #1
                               "&=",       #2
                               "\\langle", #3
                               "x",        #4
                               ",",        #5
                               "y",        #6
                               "\\rangle"  #7
                              )
        diff_simple.scale(1.6).move_to(np.array([0,-1.5,0])).align_to(diff_diff, LEFT)
        
        # x change and y change label
        x_change_label = TexMobject("\\dot x").scale(1.6).move_to(np.array([1,2.5,0])).set_color(ORANGE)
        y_change_label = TexMobject("\\dot y").scale(1.6).move_to(np.array([3.5,2.5,0])).set_color(PURPLE)
    
        screen_grid = ScreenGrid()
        self.add(screen_grid)
        
        self.wait(10)
        
        self.play(Write(x_change),
                  Write(y_change))
        self.wait(13)
        
        
        # transform differential equation into different notation
        self.play(ReplacementTransform(x_change[2].copy(), diff_diff[4]),
                  ReplacementTransform(x_change[0].copy(), x_change_label),
                  FadeOut(x_change),
                  ReplacementTransform(y_change[2].copy(), diff_diff[6]),
                  ReplacementTransform(y_change[0].copy(), y_change_label),
                  FadeOut(y_change),
                  Write(diff_diff[0:4]),
                  Write(diff_diff[5]),
                  Write(diff_diff[7]))
        self.wait(20)
        
        
        # transform into simple differential equation
        self.play(ReplacementTransform(diff_diff.copy(), diff_simple))
        self.wait(8)
        
        
     
        
        vector_field_simple = VectorField(lambda p: simple_two_d(p),
                                          # scale distance between vectors
                                          delta_x =  0.5, 
                                          delta_y =  0.5,
                                          
                                          # change magnititude range
                                          min_magnitude = 0,
                                          max_magnitude = 4,
                                          
                                          # adjust vectors
                                          vector_config = {
                                                           "stroke_width": 1,     
                                                           # "buff": MED_SMALL_BUFF, # 0.25
                                                           "max_tip_length_to_length_ratio": 0.1,
                                                           # "max_tip_length_to_length_ratio": 0.35,
                                                           # "max_stroke_width_to_length_ratio": 5,
                                                           # "max_stroke_width_to_length_ratio": 8,
                                                           # "preserve_tip_size_when_scaling": True,
                                                           # "preserve_tip_size_when_scaling": False,
                                                          },
                                          length_func = linear)#lambda norm: 1 * sigmoid(norm))
       
        # put point(2,-1) in vector field
        point = Dot([2,-1,0], color="#d8dc1a")
        
        diff_filled = TexMobject(
                               "V",        #0
                               "(2,-1)",   #1
                               "&=",       #2
                               "\\langle", #3
                               "2",        #4
                               ",",        #5
                               "-1",       #6
                               "\\rangle"  #7
                              ).scale(1.6).move_to(np.array([0,-1.5,0])).align_to(diff_diff, LEFT)
        
        self.play(ReplacementTransform(diff_simple.copy(), diff_filled),
                  FadeOut(diff_simple))
        self.play(GrowFromPoint(point, np.array([2,-1,0])))
        self.wait(5)
        
                 
    
        # display vector at one position
        vector = Vector(np.array([2,-1,0]), max_tip_length_to_length_ratio =  0.1)
        vector.move_to(np.array([3,-1.5,0])).set_stroke("#d8dc1a", 1).set_color("#d8dc1a")
        self.play(DrawBorderThenFill(vector))
        self.wait(5)
        
        
        # replace the point with general x and y
        self.play(ReplacementTransform(diff_filled.copy(), diff_simple),
                  FadeOut(diff_filled))
        
        self.wait(1)
        
        # draw vector field
        self.play(FadeOut(point), DrawBorderThenFill(vector_field_simple))
        self.wait(3)
        
        
        
        
        # fade out vector field, vector, general equation and change labels
        self.play(
                  FadeOut(vector),
                  FadeOut(vector_field_simple),
                  FadeOut(diff_diff),
                  FadeOut(x_change_label),
                  FadeOut(y_change_label)
                 )
        self.wait(2)
        
        # show different vector field
        diff_advanced = TexMobject(
                               "V",        #0
                               "(x,y)",    #1
                               "&=",       #2
                               "\\langle", #3
                               "{x+y",     #4
                               "\\over",
                               "2}"
                               ",",        #5
                               "{x-y",     #6
                               "\\over",
                               "4}",
                               "\\rangle"  #7
                              )
        diff_advanced.scale(1.6).move_to(np.array([0,-0.97,0])).align_to(diff_diff, LEFT)
        
        self.play(ReplacementTransform(diff_simple.copy(), diff_advanced),
                  FadeOut(diff_simple))
        self.wait(5)
        
        vector_field_advanced = VectorField(
                                            lambda p: advanced_two_d(p),
                                            # scale distance between vectors
                                            delta_x =  0.5, 
                                            delta_y =  0.5,
                                          
                                            # change magnititude range
                                            min_magnitude = 0,
                                            max_magnitude = 4,
                                          
                                            # adjust vectors
                                            vector_config = {
                                                           "stroke_width": 1,     
                                                           # "buff": MED_SMALL_BUFF, # 0.25
                                                           "max_tip_length_to_length_ratio": 0.1,
                                                          },
                                            length_func = linear
                                           )
        
        # draw vector field
        self.play(DrawBorderThenFill(vector_field_advanced)
                  #FadeOut(diff_advanced),
                 )
        self.remove(diff_advanced)
        self.add(diff_advanced)
        self.wait(3)
        
        
        
    
        
        
        # spiral differential equation
        
        #vector_field = VectorField(lambda p: functioncurlreal(p,0.2),
        #                           length_func = lambda norm: 0.4 * sigmoid(norm))
        
        dot1 = Dot([-1.5,3,0], color=WHITE).scale(1.5)
        dot2 = Dot([2,2,0], color=BLUE)
        self.play(GrowFromPoint(dot1, np.array([-1.5,3,0])))
        self.wait()
        
        # setup the path that follows the dot moving in the vector field
        
        dot1.start = dot1.copy
        
        # Pen
        pen = Dot(dot1.get_center(),color=WHITE)
        # Line
        line = Line(dot1.get_center(),pen.get_center()).set_stroke(BLACK,2.5)
        # Path
        path = VMobject(color=WHITE)
        # Path can't have the same coord twice, so we have to dummy point
        path.set_points_as_corners([pen.get_center(),pen.get_center()+UP*0.001])
        # Path group
        path_group = VGroup(line,pen,path)
        # Alpha, from 0 to 1:
        alpha = ValueTracker(0)
        
        
        # update function of path_group
        def update_group(group):
            l,mob,previus_path = group
            mob.move_to(dot1.get_center())
            old_path = path.copy()
            # See manimlib/mobject/types/vectorized_mobject.py
            old_path.append_vectorized_mobject(Line(old_path.points[-1],mob.get_center()))
            old_path.make_smooth()
            l.put_start_and_end_on(dot1.get_center(),mob.get_center()+UP*0.001)
            path.become(old_path)

        
        self.play(ShowCreation(line))
    
        path_group.add_updater(update_group)
        self.add(path_group)

        for dot in dot1,dot2:
            move_submobjects_along_vector_field(
                dot,
                lambda p: advanced_two_d(p)
            )
        self.wait(7)
        for dot in dot1,dot2:
            dot.clear_updaters()
        self.wait()
                  
        #c2.clear_updaters()
        path_group.clear_updaters()
        self.play(
                  FadeOut(vector_field_advanced)
                 )
        self.wait(4)
        
        
        self.play(FadeOut(diff_advanced))
        x_c_copy.move_to(np.array([0,-0.96,0]))
        y_c_copy.move_to(np.array([0,-2.84,0]))
        self.play(Write(x_c_copy),
                  Write(y_c_copy))
        
        
        
        self.wait(30)
        
        
        
        self.play(FadeOut(VGroup(path_group)),
                  FadeOut(screen_grid),
                  FadeOut(x_c_copy),
                  FadeOut(y_c_copy))
        self.wait()
        
        
        
        
        
        
        
    
    
    
#Parameters in (a): 
C = 1
I = 0              # original: 0
EL = -80
gL = 8
gNa = 20
gK = 10

mV12 = -20  
mk = 15,

nV12 = -25   
nk = 5,

τ = 1   # τ(V)=1
ENa = 60
EK = -90
    
# Inap IK differential equation
# CV = I−gL(V−EL)−gNam∞(V)(V−ENa)−gKn(V−EK)
# n ̇ = (n∞(V ) − n)/τ(V )
        

def activation(V12, k, V):
    return 1 / (1 + np.exp((V12 - V)/k))
 
    
        
def InapIK(position):
    
    V = position[0]
    n = position[1]
    
    #test to scale position so that the fit in frame
    #position[0] = position[0]/7.5 + 4.3
    #position[1] = position[1]*10 - 3
    
    # another test to scale
    V = (V*7 - 40)
    n = (n/10 + 0.4)
    
    
    V_change = I - gL*(V-EL) - gNa*activation(mV12, mk, V)*(V-ENa) - gK*n*(V-EK)
    n_change = (activation(nV12, nk, V) - n)/τ
    
    # test to scale the change a bit down so that its less red
    V_change /= 200   # V change is scaled down
    #n_change /= 40
    
    
    return V_change * RIGHT + n_change * UP     
  
    
    
    
class VectorField_InapIK(GraphScene):
    
    def construct(self):
        
        # name of model
        Inap_plus_IK = TextMobject("I Na,p + Ik - Model").scale(2)
        Inap_plus_IK.move_to(np.array([0,2,0]))
        self.play(Write(Inap_plus_IK))
        
        # show differential equation
        
        Inap_diff_V = TexMobject(
                               "C \\dot V &= I - gL \\cdot (", 
                               "V", 
                               "-EL) - gNa \\cdot m \\infty (", 
                               "V", 
                               ")(", 
                               "V", 
                               "-ENa)-gKn(",
                               "V",
                               "-EK)"
                              )
        
        Inap_diff_V.move_to(np.array([0,1,0]))
        Inap_diff_V.scale(0.75)
        
        Inap_diff_n = TexMobject("n", 
                                 "&= (n \\infty (",
                                 "V",
                                 ")-",
                                 "n",
                                 ")/ \\tau (",
                                 "V",
                                 ")")
        
        Inap_diff_n.align_to(Inap_diff_V, LEFT)
        Inap_diff_n.shift(np.array([-0.25,0.5,0]))
        Inap_diff_n.scale(0.75)
        
        self.play(Write(Inap_diff_V))
        self.wait(2)
        self.play(Write(Inap_diff_n))
        self.wait(2)
        
        
        #TODO
        #highlight the V, and n's in equation and in axis labels
        
        #my_plane = NumberPlane()
        #my_plane.add(my_plane.get_axis_labels())
        #self.add(my_plane)
        
        
        #self.setup_axes(animate=True)
        #self.wait()
        
        #screen_grid = ScreenGrid()
        #self.add(screen_grid)
        #self.wait()
        
        axes_x = Line(np.array([-7,0,0]), np.array([7,0,0]))
        axes_y = Line(np.array([0,-4,0]), np.array([0,4,0]))
        axes = VGroup(axes_x, axes_y).set_stroke(WHITE, 3)
        
        self.play(GrowFromPoint(axes, np.array([0,0,0])))
        
        x_axis_label = TexMobject("V")
        x_axis_label.move_to(np.array([6.5,0.5,0]))
        
        y_axis_label = TexMobject("n")
        y_axis_label.move_to(np.array([0.5,3.25,0]))
        
        self.play(Write(x_axis_label))
        self.play(Write(y_axis_label))
        
        
        self.play(FadeOut(Inap_diff_V),
                  FadeOut(Inap_diff_n),
                  FadeOut(Inap_plus_IK))

        
        vector_field = VectorField(lambda p: InapIK(p),
                                   length_func = lambda norm: 0.3 * sigmoid(norm),
                                   x_min = -7,
                                   x_max = 7,
                                   y_min = -4,
                                   y_max = 4,
                                   delta_x = 0.35, #vector arrow frequency
                                   delta_y = 0.2   #vector arrow frequency
                                  )
        
        
        self.play(DrawBorderThenFill(vector_field))#,dot2)
        self.wait(2)
        
        dot1 = Dot([4,-3,0], color=WHITE)
       
        
        self.play(GrowFromPoint(dot1, dot1.get_center()))
        
        
        dot1.start = dot1.copy
        
        # Pen
        pen = Dot(dot1.get_center(),color=WHITE)
        # Line
        line = Line(dot1.get_center(),pen.get_center()).set_stroke(BLACK,2.5)
        # Path
        path = VMobject(color=WHITE)
        # Path can't have the same coord twice, so we have to dummy point
        path.set_points_as_corners([pen.get_center(),pen.get_center()+UP*0.001])
        # Path group
        path_group = VGroup(line,pen,path)
        # Alpha, from 0 to 1:
        alpha = ValueTracker(0)
        
        
        # update function of path_group
        def update_group(group):
            l,mob,previus_path = group
            mob.move_to(dot1.get_center())
            old_path = path.copy()
            # See manimlib/mobject/types/vectorized_mobject.py
            old_path.append_vectorized_mobject(Line(old_path.points[-1],mob.get_center()))
            old_path.make_smooth()
            l.put_start_and_end_on(dot1.get_center(),mob.get_center()+UP*0.001)
            path.become(old_path)

        
        self.play(ShowCreation(line))
    
        path_group.add_updater(update_group)
        self.add(path_group)


        move_submobjects_along_vector_field(dot1, lambda p: InapIK(p))
        
        self.wait(40)
        dot1.clear_updaters()
                  
        #c2.clear_updaters()
        path_group.clear_updaters()
        self.play(FadeOut(VGroup(path_group)),
                  FadeOut(vector_field))
        
        self.wait(4)
        