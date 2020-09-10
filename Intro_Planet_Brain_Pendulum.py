from manimlib.imports import *

class Pendulum(VGroup):
    CONFIG = {
        "length": 3,
        "gravity": 9.8,
        "weight_diameter": 0.5,
        "initial_theta": 0.3,
        "omega": 0,
        "damping": 0.1,
        "top_point": 2 * UP,
        "rod_style": {
            "stroke_width": 3,
            "stroke_color": LIGHT_GREY,
            "sheen_direction": UP,
            "sheen_factor": 1,
        },
        "weight_style": {
            "stroke_width": 0,
            "fill_opacity": 1,
            "fill_color": GREY_BROWN,
            "sheen_direction": UL,
            "sheen_factor": 0.5,
            "background_stroke_color": BLACK,
            "background_stroke_width": 3,
            "background_stroke_opacity": 0.5,
        },
        "dashed_line_config": {
            "num_dashes": 25,
            "stroke_color": WHITE,
            "stroke_width": 2,
        },
        "angle_arc_config": {
            "radius": 1,
            "stroke_color": WHITE,
            "stroke_width": 2,
        },
        "velocity_vector_config": {
            "color": RED,
        },
        "theta_label_height": 0.25,
        "set_theta_label_height_cap": False,
        "n_steps_per_frame": 100,
        "include_theta_label": True,
        "include_velocity_vector": False,
        "velocity_vector_multiple": 0.5,
        "max_velocity_vector_length_to_length_ratio": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_fixed_point()
        self.create_rod()
        self.create_weight()
        self.rotating_group = VGroup(self.rod, self.weight)
        self.create_dashed_line()
        self.create_angle_arc()
        if self.include_theta_label:
            self.add_theta_label()
        if self.include_velocity_vector:
            self.add_velocity_vector()

        self.set_theta(self.initial_theta)
        self.update()

    def create_fixed_point(self):
        self.fixed_point_tracker = VectorizedPoint(self.top_point)
        self.add(self.fixed_point_tracker)
        return self

    def create_rod(self):
        rod = self.rod = Line(UP, DOWN)
        rod.set_height(self.length)
        rod.set_style(**self.rod_style)
        rod.move_to(self.get_fixed_point(), UP)
        self.add(rod)

    def create_weight(self):
        weight = self.weight = Circle()
        weight.set_width(self.weight_diameter)
        weight.set_style(**self.weight_style)
        weight.move_to(self.rod.get_end())
        self.add(weight)

    def create_dashed_line(self):
        line = self.dashed_line = DashedLine(
            self.get_fixed_point(),
            self.get_fixed_point() + self.length * DOWN,
            **self.dashed_line_config
        )
        line.add_updater(
            lambda l: l.move_to(self.get_fixed_point(), UP)
        )
        self.add_to_back(line)

    def create_angle_arc(self):
        self.angle_arc = always_redraw(lambda: Arc(
            arc_center=self.get_fixed_point(),
            start_angle=-90 * DEGREES,
            angle=self.get_arc_angle_theta(),
            **self.angle_arc_config,
        ))
        self.add(self.angle_arc)

    def get_arc_angle_theta(self):
        # Might be changed in certain scenes
        return self.get_theta()

    def add_velocity_vector(self):
        def make_vector():
            omega = self.get_omega()
            theta = self.get_theta()
            mvlr = self.max_velocity_vector_length_to_length_ratio
            max_len = mvlr * self.rod.get_length()
            vvm = self.velocity_vector_multiple
            multiple = np.clip(
                vvm * omega, -max_len, max_len
            )
            vector = Vector(
                multiple * RIGHT,
                **self.velocity_vector_config,
            )
            vector.rotate(theta, about_point=ORIGIN)
            vector.shift(self.rod.get_end())
            return vector

        self.velocity_vector = always_redraw(make_vector)
        self.add(self.velocity_vector)
        return self

    def add_theta_label(self):
        self.theta_label = always_redraw(self.get_label)
        self.add(self.theta_label)

    def get_label(self):
        label = TexMobject("\\theta")
        label.set_height(self.theta_label_height)
        if self.set_theta_label_height_cap:
            max_height = self.angle_arc.get_width()
            if label.get_height() > max_height:
                label.set_height(max_height)
        top = self.get_fixed_point()
        arc_center = self.angle_arc.point_from_proportion(0.5)
        vect = arc_center - top
        norm = get_norm(vect)
        vect = normalize(vect) * (norm + self.theta_label_height)
        label.move_to(top + vect)
        
        # save globally so it can me morphed into the position function theta(t) 
        # for the differential equation
        self.pendulum_theta = label
        
        return label

    #
    def get_theta(self):
        theta = self.rod.get_angle() - self.dashed_line.get_angle()
        theta = (theta + PI) % TAU - PI
        return theta

    def set_theta(self, theta):
        self.rotating_group.rotate(
            theta - self.get_theta()
        )
        self.rotating_group.shift(
            self.get_fixed_point() - self.rod.get_start(),
        )
        return self

    def get_omega(self):
        return self.omega

    def set_omega(self, omega):
        self.omega = omega
        return self

    def get_fixed_point(self):
        return self.fixed_point_tracker.get_location()

    #
    def start_swinging(self):
        self.add_updater(Pendulum.update_by_gravity)

    def end_swinging(self):
        self.remove_updater(Pendulum.update_by_gravity)

    def update_by_gravity(self, dt):
        theta = self.get_theta()
        omega = self.get_omega()
        nspf = self.n_steps_per_frame
        for x in range(nspf):
            d_theta = omega * dt / nspf
            d_omega = op.add(
                -self.damping * omega,
                -(self.gravity / self.length) * np.sin(theta),
            ) * dt / nspf
            theta += d_theta
            omega += d_omega
        self.set_theta(theta)
        self.set_omega(omega)
        return self



class ThetaValueDisplay(VGroup):
    CONFIG = {

    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ThetaVsTAxes(Axes):
    CONFIG = {
        "x_min": 0,
        "x_max": 8,
        "y_min": -PI / 2,
        "y_max": PI / 2,
        "y_axis_config": {
            "tick_frequency": PI / 8,
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
        self.add(self.axes)

    def add_labels(self):
        x_axis = self.get_x_axis()
        y_axis = self.get_y_axis()

        t_label = self.t_label = TexMobject("t")
        t_label.next_to(x_axis.get_right(), UP, MED_SMALL_BUFF)
        x_axis.label = t_label
        x_axis.add(t_label)
        theta_label = self.theta_label = TexMobject("\\theta(t)")
        theta_label.next_to(y_axis.get_top(), UP, SMALL_BUFF)
        #y_axis.label = theta_label
        #y_axis.add(theta_label)

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
            "\\pi / 4",
            "\\pi / 2",
            "3 \\pi / 4",
            "\\pi",
        ]
        values = np.arange(1, 5) * PI / 4
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

    def get_live_drawn_graph(self, pendulum,
                             t_max=None,
                             t_step=1.0 / 60,
                             **style):
        style = merge_dicts_recursively(self.graph_style, style)
        if t_max is None:
            t_max = self.x_max

        graph = VMobject()
        graph.set_style(**style)

        graph.all_coords = [(0, pendulum.get_theta())]
        graph.time = 0
        graph.time_of_last_addition = 0

        def update_graph(graph, dt):
            graph.time += dt
            if graph.time > t_max:
                graph.remove_updater(update_graph)
                return
            new_coords = (graph.time, pendulum.get_theta())
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


class GravityVector(Vector):
    CONFIG = {
        "color": ORANGE,
        # original: 1/9.8
        "length_multiple": 2 / 9.8,
        # TODO, continually update the length based
        # on the pendulum's gravity?
    }

    def __init__(self, pendulum, **kwargs):
        super().__init__(DOWN, **kwargs)
        self.pendulum = pendulum
        self.scale(self.length_multiple * pendulum.gravity)
        self.attach_to_pendulum(pendulum)

    def attach_to_pendulum(self, pendulum):
        self.add_updater(lambda m: m.shift(
            pendulum.weight.get_center() - self.get_start(),
        ))

    def add_component_lines(self):
        self.component_lines = always_redraw(self.create_component_lines)
        self.add(self.component_lines)

    def create_component_lines(self):
        theta = self.pendulum.get_theta()
        x_new = rotate(RIGHT, theta)
        base = self.get_start()
        tip = self.get_end()
        vect = tip - base
        corner = base + x_new * np.dot(vect, x_new)
        kw = {"dash_length": 0.025}
        return VGroup(
            DashedLine(base, corner, **kw),
            DashedLine(corner, tip, **kw),
        )




#HSL color, see https://pypi.org/project/colour/
def HSL(hue,saturation=1,lightness=0.5):
    return Color(hsl=(hue,saturation,lightness))


# This function is come and go, but linear
def double_linear(t):
    if t < 0.5:
        return linear(t*2)
    else:
        return linear(1-(t-0.5)*2)
    

class Planet_Brain_Pendulum(Scene):
    
    def highlight_object(self, thing, size = 1.5):
        self.play(ApplyMethod(thing.scale, size))
        self.play(ApplyMethod(thing.scale, 1/size))
    
    CONFIG = {
        # brain activity config
        "number_of_lines": 200,
        "gradient_colors":[RED,YELLOW,BLUE],
        # set speed of brain activity
        #"start_value":0,
        "end_value":20,
        "total_time":6,
        
        # pendulum config
        "pendulum_config": {
            "initial_theta": 35 * DEGREES,
            "length": 2.0,
            "damping": 0,
            # pendulum position
            "top_point": np.array([0,-1.0,0]),
        },
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
        },
        "axes_corner": UL,
    }
    
    def get_m_mod_n_objects(self,circle,x,y=None):
        if y==None:
            y = self.number_of_lines
        lines = VGroup()
        for i in range(y):
            start_point = circle.point_from_proportion((i%y)/y)
            end_point = circle.point_from_proportion(((i*x)%y)/y)
            line = Line(start_point,end_point).set_stroke(width=1)
            lines.add(line)
        lines.set_color_by_gradient(*self.gradient_colors)
        return lines

    
    
    def construct(self):
        
        # setup stuff
        self.period_formula = TexMobject(
            "2\\pi", "\\sqrt{\\,", "L", "/", "g", "}",
            tex_to_color_map={
                "L": BLUE,
                "g": ORANGE,
            }
        )
        
        # TODO
        # start brain activity in a messy state
        
        # create brain activity
        circle = Circle().set_height(FRAME_HEIGHT*0.135).move_to(np.array([3.10,.97,0]))   # brain size and position
        mod_tracker = ValueTracker(0)
        lines = self.get_m_mod_n_objects(circle,mod_tracker.get_value())
        lines.add_updater(
            lambda mob: mob.become(
                self.get_m_mod_n_objects(circle,mod_tracker.get_value())
                )
            )
       
        # vary size of planetary system
        s = 0.5 
        
        
        # create Planet System
        sun = Dot(point = np.array([-3,1,0]), radius = 0.4*s)
        sun.set_color(YELLOW)
        
        planet_1 = Dot(point = np.array([-3,3*s,0]), radius = 0.2*s, color = ORANGE)
        planet_2 = Dot(point = np.array([-3,4*s,0]), radius = 0.2*s, color = GREEN)
        planet_3 = Dot(point = np.array([-3,5*s,0]), radius = 0.2*s, color = BLUE)
        
        orbit_1 = Circle(arc_center = np.array([-3,1,0]), radius = 1*s, color = ORANGE)
        orbit_2 = Circle(arc_center = np.array([-3,1,0]), radius = 2*s, color = GREEN)
        orbit_3 = Circle(arc_center = np.array([-3,1,0]), radius = 3*s, color = BLUE)
        
        # create pendulum
        pendulum = Pendulum(**self.pendulum_config)
        self.pendulum = pendulum
        
        
        # create Brain
        brain = ImageMobject('images/brain_img_white.png')
        brain.scale(1)  # Resize 
        brain.shift(np.array([3,1,0]))  # Move the image
        
        self.wait(2)
        
        # show objects
        self.play(GrowFromPoint(sun, point = np.array([-3,1,0])),
                  GrowFromPoint(orbit_1, point = np.array([-3,1,0])),
                  GrowFromPoint(orbit_2, point = np.array([-3,1,0])),
                  GrowFromPoint(orbit_3, point = np.array([-3,1,0])),
                 )
        
        self.play(AnimationGroup(*[GrowFromPoint(planet_1, point = planet_1.arc_center),
                                   GrowFromPoint(planet_2, point = planet_2.arc_center),
                                   GrowFromPoint(planet_3, point = planet_3.arc_center)], lag_ratio=0.04))
        
        # wait for script
        #self.wait(0.2)
        
        # add brain and brain activity
        self.play(AnimationGroup(*[GrowFromPoint(brain, point = np.array([3,1,0])),
                                   ShowCreation(lines)], lag_ratio = 0.4))
        
        self.wait(0.4)
        self.play(GrowFromPoint(pendulum, point = np.array([0,-1,0])))
        
        # wait for script
        self.wait(9)
        
        # pendulum swinging
        pendulum.start_swinging()
        
        # rotate planets
        self.play(# planet rotation
                  Rotating(planet_1, radians=4 * TAU, about_point=sun.arc_center, rate_func = smooth, run_time = 6),
                  Rotating(planet_2, radians=3 * TAU, about_point=sun.arc_center, rate_func = smooth, run_time = 6),
                  Rotating(planet_3, radians=2 * TAU, about_point=sun.arc_center, rate_func = smooth, run_time = 6),
                  # brain activity
                  mod_tracker.set_value,self.end_value,
                  rate_func=linear,
                  run_time=self.total_time)
        
        # stop pendulum
        self.wait(0.5)
        pendulum.set_theta(0)
        pendulum.end_swinging()
       
        
        self.wait(5)
        
        
        
        
        
        # zoom in on pendulum, everything else is removed
        self.play(FadeOut(lines))
        self.play(FadeOut(sun), 
                  FadeOut(planet_1), FadeOut(planet_2), FadeOut(planet_3),
                  FadeOut(orbit_1), FadeOut(orbit_2), FadeOut(orbit_3),
                  FadeOut(brain))
        
        
        self.wait(3)
        # move og pendulum
        self.play(ApplyMethod(pendulum.shift, np.array([0,1.5,0])))
        self.move_pendulum_to_angle(pendulum, angle = 35)
       
        
    
        
        
        # introduce 2 new pendulums one with different string length and one with different gravity
        # swing them in comparison to the orignial pendulum
        
        pendulum_2 = Pendulum(**{
            "initial_theta": 35 * DEGREES,
            "length": 1,
            "damping": 0,
            # pendulum position
            "top_point": np.array([-4,0.5,0]),
        })
        
        # gravity and length label
        g_1 = TexMobject("gravity: \\; 9.8").set_color(ORANGE)
        L_1 = TexMobject("length: \\; 2").set_color(GREEN)
        
        L_2 = TexMobject("length: \\; 1").set_color(GREEN)
        
        g_3 = TexMobject("gravity: \\; 3").set_color(ORANGE)
    
        # set position of labels
        g_1.move_to(np.array([0,1.5,0]))
        L_1.move_to(np.array([0,2,0]))
        L_2.move_to(np.array([-4,2,0]))
        g_3.move_to(np.array([4,1.5,0]))
        
        self.play(Write(g_1),
                  Write(L_1))
        
        self.wait(2)
        
        self.play(GrowFromPoint(pendulum_2, point = np.array([0,-1,0])),
                  Write(L_2))
        pendulum.set_theta(35 * DEGREES)
        pendulum.set_omega(0)
        self.wait(11)
        
        
        
        #pendulum.generate_target()
        #pendulum.target.set_theta(35 * DEGREES)
        #self.play(MoveToTarget(pendulum),
        #          run_time = 0.5)
        # 
        #self.wait(1)
        
        pendulum.start_swinging()
        pendulum_2.start_swinging()
        
        
        
        
        
        pendulum_3 = Pendulum(**{
            "initial_theta": 35 * DEGREES,
            "length": 2,
            "damping": 0,
            "gravity": 3,
            # pendulum position
            "top_point": np.array([4,0.5,0]),
        })
        
        self.play(GrowFromPoint(pendulum_3, point = np.array([0,-1,0])),
                  Write(g_3))
        #self.move_pendulum_to_angle(pendulum, angle = 35)
        self.wait(2)
        
        
        
        
        #pendulum.generate_target()
        #pendulum.target.set_theta(35 * DEGREES)
        #self.play(MoveToTarget(pendulum),
        #          run_time = 0.5)
        # 
        #self.wait(1)
        
        #pendulum.start_swinging()
        pendulum_3.start_swinging()
        self.wait(22)
        pendulum.end_swinging()
        pendulum_2.end_swinging()
        pendulum_3.end_swinging()

        
        self.move_pendulum_to_angle(pendulum, angle = 0) 
        
        self.play(FadeOut(pendulum_2),
                  FadeOut(pendulum_3),
                  FadeOut(g_1), 
                  FadeOut(L_1),
                  FadeOut(L_2),
                  FadeOut(g_3)
                 )

        
        
        
        
        
        
        

        self.move_pendulum_to_angle(pendulum, angle = 35)
        
        
        
        
        # explain pendulum accelartion
        
        
        
        
        
        
        
        
        
        
        # explaining the pendulums angle changes with differential equations
        
        #pend_eq_1 = TexMobject(r"\theta").scale(2).move_to(np.array([0,1,0]))
        #self.play(Write(pend_eq_1))
        #self.wait(2)
        #self.play(ApplyMethod(pend_eq_1.shift, np.array([-6,0,0])))
        
        # graph of theta evolving as pendulum swings
        axes = ThetaVsTAxes(**self.axes_config)
        self.axes = axes
        #axes.center()
        #axes.to_corner(self.axes_corner, buff=LARGE_BUFF)
        axes.move_to(np.array([0,1,0]))
        self.add(axes)
        self.wait(9)
        
        graph = axes.get_live_drawn_graph(pendulum)
        pendulum.start_swinging()
        self.add(graph)
        self.wait(15)
        
        # add theta(0)?
        
        position_function = TexMobject("? \\quad \\theta (t) \\quad ?").move_to(np.array([0,2.5,0])).set_color(BLUE).scale(2)
        self.play(Write(position_function))
        self.wait(13)
        
        self.remove(graph)
        self.play(FadeOut(position_function))
        
        
        
        #pendulum = Pendulum(**self.pendulum_config_2)       # in case we need a new pendulum

        
        # remove pendulum, scale graph, move graph in center
        pendulum.end_swinging()
        self.wait()
        
        self.play(FadeOut(pendulum), 
                  ApplyMethod(axes.shift, np.array([0,-1,0]))) 
        
        self.play(ApplyMethod(self.axes.scale, 1.2))
        
        
        #pend_eq_2 = TexMobject("\\dot{", "\\theta}",).scale(2).move_to(np.array([0,2,0]))
        #self.play(Transform(pend_eq_1, pend_eq_2))
        #self.wait(2)
        
        
        # self.play(ApplyMethod(axes.shift, np.array([0,-2,0]))) 
        
        # differential equation that describes the pendulum movement
        pend_equation = TexMobject("\\ddot \\theta", #0
                                   "(t)",            #1
                                   "&=",             #2
                                   "-",              #3
                                   "{g",             #4
                                   "\\over",         #5
                                   "L}",             #6 
                                   "sin(",           #7
                                   "\\theta",        #8
                                   "(t)",            #9
                                   ")")              #10
        pend_equation.scale(2)
        pend_equation.move_to(np.array([0,2,0]))
        
        pend_equation[0].set_color(YELLOW)
        pend_equation[8].set_color(BLUE)
        
        # save equation globally so that it runs simultaniously with the graph
        self.pend_equation = pend_equation
        
        # add labels
        accel_label = TexMobject("Acceleration").move_to(np.array([-3,3.3,0])).scale(0.6).set_color(YELLOW)
        pos_label = TexMobject("Position").move_to(np.array([3,3.3,0])).scale(0.6).set_color(BLUE)
       
        
        self.accel = accel_label
        self.pos = pos_label
    
        
        
        # start graph animation of position, speed and acceleration
        self.show_value_slope_curvature()# (scaler = 1)
        self.wait(13)
        self.show_changing_curvature_group()
        #self.wait(2)
        
        
        # show differential equation which describes releation of position and acceleration
        self.play(Write(self.pend_equation))
        self.play(Write(self.accel))
        self.play(Write(self.pos))
        self.wait(1)
        
        
        # remove axes and dahed_graph
        self.play(FadeOut(axes),
                  FadeOut(self.dashed_graph),
                  FadeOut(self.curvature_group))
        
        
        # add pendulum which explains parts of the equation
        pendulum = Pendulum(**{
            "initial_theta": 35 * DEGREES,
            "length": 2.5,
            "damping": 0,
            # pendulum position
            "top_point": np.array([0,0.0,0]),
        })
        self.pendulum = pendulum
        pendulum.set_theta(math.pi/6)
        self.play(GrowFromPoint(pendulum, point = np.array([0,1,0])))
        self.wait(4)
        
        # premade length and gravity showcase
        self.show_length_and_gravity()
        
        
    
        # remove Pendulum
        self.play(FadeOut(self.pendulum))
        
        
        
        # add axes and dahed_graph
        self.play(FadeIn(axes),
                  FadeIn(self.dashed_graph),
                  FadeIn(self.curvature_group))
        
        
        self.wait(6.5)
        
        
        # highlight path
        self.dashed_graph.target.set_stroke(BLUE, 12)
        self.play(MoveToTarget(self.dashed_graph),
                  run_time = 1)
        self.wait(1)
        self.dashed_graph.target.set_stroke(BLUE, 3)
        self.play(MoveToTarget(self.dashed_graph),
                  run_time = 1)
        self.wait(14)
        
    
        
        # highlight input and output
        self.highlight_object(self.pend_equation[8:10], 1.5)
        self.wait(0.5)
        self.highlight_object(self.pend_equation[0:2], 1.5)


        
        # add path as an estimation of the actual path or solution of the differential equation
        # first few and big vectors that wander of the path, then many small ones that follow it somewhat acuratly
        # Pendulum setup
        # Pendulum setup
        gravity = 9.8                 # in m/s^2
        length = 1                    # in m
        initial_angle = 45*DEGREES    # in radian

        def coord(x,y,z=0):
            return np.array([x,y,z])
            
        def pend_accel(angle):
            accel = -gravity/length * math.sin(angle)
            angle_accel = accel/length                  # angle accel in radian
            return angle_accel


        def get_next_angle(angle, step_size, current_speed):
            average_speed = current_speed + (pend_accel(angle)*step_size)/2
            angle_change = average_speed*step_size

            current_speed += pend_accel(angle)*step_size
            next_angle = angle + angle_change

            return (next_angle, current_speed)



        def get_points(step_count = 10, step_size = 0.1):

            # set current speed to zero before the pendulum is let off
            current_speed = 0           # in m/s
            current_time = 0            # in s
            current_angle = initial_angle


            points = [(current_time, current_angle)]
            for i in range(step_count):
                current_time += step_size
                current_angle, current_speed =  get_next_angle(current_angle, step_size, current_speed)
                points.append((current_time, current_angle))
            return points


        self.wait(19.5)
        
        
        # create a good path with 11000 points and a step_size of 0.001 seconds
        points_for_path = get_points(11000, 0.001)
        path = VMobject()
        path.set_points_as_corners([*[coord(x,y) for x,y in points_for_path]])
        path.shift(np.array([-4.5,0,0]))
        self.play(FadeIn(path))
        self.wait(44.5)
        self.play(FadeOut(path))
        
        
        # create a bad path with 110 points and a step_size of 0.1 seconds
        points_for_path_2 = get_points(110, 0.1)
        path_2 = VMobject()
        path_2.set_points_as_corners([*[coord(x,y) for x,y in points_for_path_2]])
        path_2.shift(np.array([-4.5,0,0]))
        self.play(FadeIn(path_2))
        
        # add points which make up path
        #for x,y in points_for_path:
            
        
        self.wait(20)
        self.play(FadeOut(path_2),
                  FadeOut(axes),
                  FadeOut(self.dashed_graph),
                  FadeOut(self.curvature_group))
        self.wait(26)
        
       

        # show solution of differential equation
        self.show_diff_solution()
        self.wait(16)
        
        
        
        
        # transition to the general form of differntial equations of that type (independent of time)
        diff_equation = TexMobject("\\dot y", #0
                                   "(t)",     #1
                                   "&=",      #2
                                   "f(",      #3
                                   "y",       #4
                                   "(t))"     #5
                                  ).scale(2)    
                
        diff_equation.move_to(np.array([0,0,0]))
        diff_equation.align_to(pend_equation, LEFT)
        
        diff_equation[0].set_color(YELLOW)
        diff_equation[4].set_color(BLUE)
        
        
        self.play(ReplacementTransform(pend_equation[3:10].copy(), diff_equation[3:6]))
        self.play(ReplacementTransform(pend_equation[0:3].copy(), diff_equation[0:3]))
        self.wait(6)
        
        
        # Show the general form for all types
        diff_equation_general = TexMobject("\\dot y", #0
                                           "(t)",     #1
                                           "&=",      #2
                                           "f(",      #3
                                           "y",       #4
                                           "(t), t)"  #5
                                  ).scale(2)    
        
        diff_equation_general.move_to(np.array([0,0,0]))
        diff_equation_general.align_to(pend_equation, LEFT)
        
        diff_equation_general_short = TexMobject("\\dot y", #0
                                                 "&=",      #1
                                                 "f(",      #2
                                                 "y",       #3
                                                 ", t)"     #4
                                  ).scale(2)    
        
        diff_equation_general_short.move_to(np.array([0,-2,0]))
        diff_equation_general_short.align_to(pend_equation, LEFT)
        diff_equation_general_short[0].set_color(YELLOW)
        diff_equation_general_short[3].set_color(BLUE)
        
        self.play(Transform(diff_equation[5], diff_equation_general[5]))
        self.wait(28)
        
        self.play(ReplacementTransform(diff_equation.copy(), diff_equation_general_short))
        self.wait(10)
        
        
        

      
    def move_pendulum_to_angle(self, pendulum, angle = 0):
    
        # add theta_tracker
        # it's important to only add this here because it messes with the pendulum otherwise
        pendulum.set_theta(0)
        pendulum.set_omega(0)
        theta_tracker = ValueTracker(pendulum.get_theta())
        updater = lambda p: p.set_theta(theta_tracker.get_value())
        pendulum.add_updater(updater)
        self.theta_tracker = theta_tracker
        
        self.play(
            self.theta_tracker.increment_value, angle*DEGREES,                
        )
        
        # remove updater so that the pendulum can swing freely again
        pendulum.remove_updater(updater) 

    # function to move pendulum arbitrarily
    def show_constraint(self):
        pendulum = self.pendulum
        
        # add theta_tracker
        # it's important to only add this here because it messes with the pendulum otherwise
        theta_tracker = ValueTracker(pendulum.get_theta())
        updater = lambda p: p.set_theta(theta_tracker.get_value())
        pendulum.add_updater(updater)
        self.theta_tracker = theta_tracker
        
        #arcs = VGroup()
        #for u in [-1, 2, -2]:   # [-1, 2, -1]
        #    d_theta = 100 * DEGREES * u
        #    arc = Arc(
        #        start_angle=pendulum.get_theta() - 90 * DEGREES,
        #        angle=d_theta,
        #        radius=pendulum.length,
        #        arc_center=pendulum.get_fixed_point(),
        #        stroke_width=2,
        #        stroke_color=YELLOW,
        #        stroke_opacity=0.5,
        #    )
        self.play(
            self.theta_tracker.increment_value, 35*DEGREES,
                #ShowCreation(arc)
            )
            #arcs.add(arc)
        #self.play(FadeOut(arcs))
        
        # remove updater so that the pendulum can swing freely again
        pendulum.remove_updater(updater) 
        
        
    def show_diff_solution(self):
        # show solution of differntial equation
        solution_p1 = TexMobject(
            """
            \\theta(t) = 2\\text{am}\\left(
                \\frac{\\sqrt{2g + Lc_1} (t + c_2)}{2\\sqrt{L}},
                \\frac{4g}{2g + Lc_1}
            \\right)
            """,
        )
        solution_p1.to_corner(UL)
        solution_p2 = TexMobject(
            "c_1, c_2 = \\text{Constants depending on initial conditions}"
        )
        solution_p2.set_color(LIGHT_GREY)
        solution_p2.scale(0.75)
        solution_p3 = TexMobject(
            """
            \\text{am}(u, k) =
            \\int_0^u \\text{dn}(v, k)\\,dv
            """
        )
        solution_p3.name = TextMobject(
            "(Jacobi amplitude function)"
        )
        solution_p4 = TexMobject(
            """
            \\text{dn}(u, k) =
            \\sqrt{1 - k^2 \\sin^2(\\phi)}
            """
        )
        solution_p4.name = TextMobject(
            "(Jacobi elliptic function)"
        )
        solution_p5 = TextMobject("Where $\\phi$ satisfies")
        solution_p6 = TexMobject(
            """
            u = \\int_0^\\phi \\frac{dt}{\\sqrt{1 - k^2 \\sin^2(t)}}
            """
        )

        solution = VGroup(
            solution_p1,
            solution_p2,
            solution_p3,
            solution_p4,
            solution_p5,
            solution_p6,
        )
        solution.arrange(DOWN)
        solution.scale(0.62)
        solution.move_to(np.array([0,-1.5,0]))
        solution.set_stroke(width=0, background=True)
        
        self.play(Write(solution))
        self.wait(10)
        self.play(FadeOut(solution))
        
    def show_value_slope_curvature(self, scaler = 1):
        axes = self.axes
        graph = axes.get_graph(
            lambda t: -0.90 * np.cos(1.8 * t) # * np.exp(-0.2 * t) # for dampening
        ).scale(scaler)
        
        tex_config = {
            "tex_to_color_map": {
                "{\\theta}": BLUE,
                "{\\dot \\theta}": RED,
                "{\\ddot \\theta}": YELLOW,
            },
            "height": 0.5,
        }
        x, d_x, dd_x = [
            TexMobject(
                "{" + s + "\\theta}(t)",
                **tex_config
            )
            for s in ("", "\\dot ", "\\ddot ")
        ]

        t_tracker = ValueTracker(1.25) #1.25
        get_t = t_tracker.get_value

        def get_point(t):
            return graph.point_from_proportion(t / axes.x_max)

        def get_dot():
            return Dot(get_point(get_t())).scale(0.5)

        def get_v_line():
            point = get_point(get_t())
            x_point = axes.x_axis.number_to_point(
                axes.x_axis.point_to_number(point)
            )
            return DashedLine(
                x_point, point,
                dash_length=0.025,
                stroke_color=WHITE,
                stroke_width=4, # original 2
            )

        def get_tangent_line(curve, alpha):
            line = Line(
                ORIGIN, 1.5 * RIGHT,
                color=RED,
                stroke_width=3, # orignal 1.5
            )
            da = 0.0001
            p0 = curve.point_from_proportion(alpha)
            p1 = curve.point_from_proportion(alpha - da)
            p2 = curve.point_from_proportion(alpha + da)
            angle = angle_of_vector(p2 - p1)
            line.rotate(angle)
            line.move_to(p0)
            return line

        def get_slope_line():
            return get_tangent_line(
                graph, get_t() / axes.x_max
            )

        def get_curve():
            curve = VMobject()
            t = get_t()
            curve.set_points_smoothly([
                get_point(t + a)
                for a in np.linspace(-0.5, 0.5, 11)
            ])
            curve.set_stroke(YELLOW, 2) # origianl 1
            return curve

        v_line = always_redraw(get_v_line)
        dot = always_redraw(get_dot)
        slope_line = always_redraw(get_slope_line)
        curve = always_redraw(get_curve)

        x.next_to(v_line, RIGHT, SMALL_BUFF)
        d_x.next_to(slope_line.get_end(), UP, SMALL_BUFF)
        dd_x.next_to(curve.get_end(), RIGHT, SMALL_BUFF)
        xs = VGroup(x, d_x, dd_x)

        words = VGroup(
            TextMobject("= Position").set_color(BLUE),
            TextMobject("= Speed").set_color(RED),
            TextMobject("= Acceleration").set_color(YELLOW),
        )
        words.scale(0.75)
        for word, sym in zip(words, xs):
            word.next_to(sym, RIGHT, buff=2 * SMALL_BUFF)
            sym.word = word

        self.play(
            ShowCreation(v_line),
            FadeInFromPoint(dot, v_line.get_start()),
            FadeInFrom(x, DOWN),
            FadeInFrom(x.word, DOWN),
        )
        self.add(slope_line, dot)
        self.play(
            ShowCreation(slope_line),
            FadeInFrom(d_x, LEFT),
            FadeInFrom(d_x.word, LEFT),
        )

        a_tracker = ValueTracker(0)
        curve_copy = curve.copy()
        changing_slope = always_redraw(
            lambda: get_tangent_line(
                curve_copy,
                a_tracker.get_value(),
            ).set_stroke(
                opacity=there_and_back(a_tracker.get_value())
            )
        )
        self.add(curve, dot)
        self.play(
            ShowCreation(curve),
            FadeInFrom(dd_x, LEFT),
            FadeInFrom(dd_x.word, LEFT),
        )
        self.add(changing_slope)
        self.play(
            a_tracker.set_value, 1,
            run_time=3,
        )
        self.remove(changing_slope, a_tracker)

        self.t_tracker = t_tracker
        self.curvature_group = VGroup(
            v_line, slope_line, curve, dot
        )
        self.curvature_group_labels = VGroup(xs, words)
        self.fake_graph = graph
        
   

    def show_changing_curvature_group(self):
        t_tracker = self.t_tracker
        curvature_group = self.curvature_group
        labels = self.curvature_group_labels
        graph = VMobject()
        graph.pointwise_become_partial(
            self.fake_graph,
            t_tracker.get_value() / self.axes.x_max,
            1,
        )
        dashed_graph = DashedVMobject(graph, num_dashes=100)
        dashed_graph.set_stroke(WHITE, 1)

        self.play(FadeOut(labels))
        self.add(dashed_graph, curvature_group)
        
        
        self.play(
            t_tracker.set_value, 20,     # original at 10, set to 20 so it matches graph speed
            ShowCreation(dashed_graph),
            run_time=19,                 # original at 15 set to 20 for slower animation
            rate_func=linear,
        )
        
      
        dashed_graph.generate_target()
        dashed_graph.target.set_stroke(BLUE, 8)
        self.play(MoveToTarget(dashed_graph),
                  run_time = 1)
        
        self.wait(1)
        
        dashed_graph.target.set_stroke(BLUE, 3)
        self.play(MoveToTarget(dashed_graph),
                  run_time = 1)
        
        self.dashed_graph = dashed_graph
        
        

        
            
            
        
    def show_length_and_gravity(self):
        
        L = TexMobject("L").set_color(GREEN)
        g = TexMobject("g").set_color(ORANGE)
        
        
        #weight = self.pendulum.weight
        #weight.generate_target()
        #weight.target.set_color(BLUE)
        
        theta_lab = self.pendulum.theta_label
        theta_lab.generate_target()
        theta_lab.target.set_color(BLUE)

        rod = self.pendulum.rod
        new_rod = rod.copy()
        new_rod.set_stroke(GREEN, 7)
        new_rod.add_updater(lambda r: r.put_start_and_end_on(
            *rod.get_start_and_end()
        ))

        g_vect = GravityVector(
            self.pendulum,
            # original: 0.5/9.8
            length_multiple=1 / 9.8,
        )
        down_vectors = self.get_down_vectors()
        down_vectors.set_color(ORANGE)
        down_vectors.set_opacity(0.5)

        L.next_to(new_rod)
        L.shift(np.array([-0.2,0,0]))
        g.next_to(g_vect)
        
        
        self.play(
            ApplyMethod(self.pend_equation[6].set_color, GREEN),
            Write(L),
            ShowCreation(new_rod),
            
        )
        
        
        self.play(MoveToTarget(theta_lab))
        
        self.play(
            ApplyMethod(self.pend_equation[4].set_color, ORANGE),
            Write(g),
            GrowArrow(g_vect),
        )
        self.play(self.get_down_vectors_animation(down_vectors))

        self.gravity_vector = g_vect
        
        # highlight position
        self.highlight_object(self.pend_equation[8:10], 1.5)

        
        # remove the stuff
        self.play(
            FadeOut(L),
            FadeOut(new_rod),
            FadeOut(g),
            FadeOut(g_vect),
        )

    def get_down_vectors(self):
        down_vectors = VGroup(*[
            Vector(0.5 * DOWN)
            for x in range(10 * 150)
        ])
        down_vectors.arrange_in_grid(10, 150, buff=MED_SMALL_BUFF)
        down_vectors.set_color_by_gradient(BLUE, RED)
        # for vect in down_vectors:
        #     vect.shift(0.1 * np.random.random(3))
        down_vectors.to_edge(RIGHT)
        return down_vectors
    
    def get_down_vectors_animation(self, down_vectors):
        return LaggedStart(
            *[
                GrowArrow(v, rate_func=there_and_back)
                for v in down_vectors
            ],
            lag_ratio=0.0005,
            run_time=2,
            remover=True
        )
                                
        
class Differential_Equations(Scene):
    
    def construct(self):
        
        # Position of pendulum indicated by its degree relative to normal state
        diff_1 = TexMobject(r"y").scale(2)
    
        diff_2 = TexMobject(r"y(t)").scale(2)
        diff_3 = TexMobject("y", "'", "(t) &= f(y(t))").scale(2)
        diff_4 = TexMobject("\\dot{", "y}", "(t) &= f(y(t))").scale(2)
        diff_4[0].set_color("#e71212")
        
        diff_5_simple = TexMobject("\\dot{", "y}", " &= f(y)").scale(2)
        diff_5_simple[0].set_color("#e71212")
        
        self.play(Write(diff_1))
        self.wait(2)
        self.play(Transform(diff_1, diff_2))
        self.wait(2)
        self.play(Transform(diff_1, diff_3))
        self.wait(2)
        
        self.play(ApplyMethod(diff_1[1].set_color,"#e71212"))
        self.wait(2)
        
        diff_4[0].set_color("#e71212")
        self.play(Transform(diff_1, diff_4))
        self.wait(2)
        
        self.play(Transform(diff_1, diff_5_simple))
        self.wait(2)
        
        #self.play(ApplyMethod(diff_general_1.move_to,np.array([0,3,0])))
        #self.wait(5)
        
        
class Differential_Equations_2(Scene):
    
    def construct(self):
        
        # Position of pendulum indicated by its angle relative to normal state
        diff_1 = TexMobject(r"y &= \theta").scale(2)
    
        # If we consider a swinging pendulum than its angle changes as a function of time t
        diff_2 = TexMobject(r"y(t) &= \theta").scale(2)
        
        
        diff_3 = TexMobject("y", "'", "(t) &= f(y(t))").scale(2)
        diff_4 = TexMobject("\\dot{", "y}", "(t) &= f(y(t))").scale(2)
        diff_4[0].set_color("#e71212")
        
        diff_5_simple = TexMobject("\\dot{", "y}", " &= f(y)").scale(2)
        diff_5_simple[0].set_color("#e71212")
        
        self.play(Write(diff_1))
        self.wait(2)
        self.play(Transform(diff_1, diff_2))
        self.wait(2)
        self.play(Transform(diff_1, diff_3))
        self.wait(2)
        
        self.play(ApplyMethod(diff_1[1].set_color,"#e71212"))
        self.wait(2)
        
        diff_4[0].set_color("#e71212")
        self.play(Transform(diff_1, diff_4))
        self.wait(2)
        
        self.play(Transform(diff_1, diff_5_simple))
        self.wait(2)
        
        
