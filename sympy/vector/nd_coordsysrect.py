from inspect import signature
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.basic import Basic
from sympy.core.compatibility import string_types, range, Callable
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy import symbols, MatrixBase, ImmutableDenseMatrix,\
    eye, trigsimp, ImmutableMatrix as Matrix, Symbol, sin, cos,\
    sqrt, diff, Tuple, acos, atan2, simplify
from sympy.solvers import solve
from sympy.vector.nd_scalar import BaseScalar
import sympy.vector
from sympy.vector.nd_orienters import (Orienter, AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)


        
class CoordSysND(Basic):
    """
    Represents a coordinate system in N-D space.
    """

    def __new__(cls, name, transformation=None, parent=None, location=None,
                rotation_matrix=None, vector_names=None, variable_names=None,dim=None):
        """
        The orientation/location parameters are necessary if this system
        is being defined at a certain orientation or location wrt another.

        Parameters
        ==========

        name : str
            The name of the new CoordSysND instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        location : Vector
            The position vector of the new system's origin wrt the parent
            instance.

        rotation_matrix : SymPy ImmutableMatrix
            The rotation matrix of the new coordinate system with respect
            to the parent. In other words, the output of
            new_system.rotation_matrix(parent).

        parent : CoordSysND
            The coordinate system wrt which the orientation/location
            (or both) is being defined.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        """
        ld=locals()
        dimArgNames=["parent","location","rotation_matrix","vector_names","variable_names","dim"]
        dimArgVals=[ld[argName] for argName in dimArgNames]
        if all([arg is None for arg in dimArgVals]):
           raise ValueError("Dimension could not be inferred At least one of {0} has to be present.".format(dimArgNames))

        name = str(name)
        Vector = sympy.vector.Vector
        BaseVector = sympy.vector.BaseVector
        Point = sympy.vector.Point
        n_t0=None
        n_t1=None
        if not isinstance(name, string_types):
            raise TypeError("name should be a string")

        if transformation is not None:
            if (location is not None) or (rotation_matrix is not None):
                raise ValueError("specify either `transformation` or "
                                 "`location`/`rotation_matrix`")
            if isinstance(transformation, (Tuple, tuple, list)):
                if isinstance(transformation[0], MatrixBase):
                    rotation_matrix = transformation[0]
                    location = transformation[1]
                else:
                    transformation = Lambda(transformation[0],
                                            transformation[1])
                    transformation_dim0=len(transformation[0])
                    transformation_dim1=len(transformation[1])
            elif isinstance(transformation, Callable):
                if isinstance(transformation, Lambda):
                    sbs=transformation.args[0]
                else:
                    # normal function
                    sig=signature(transformation)
                    arglist=[a for a in sig.parameters]                
                    sbs=tuple(symbols(arglist,cls=Dummy))
                #x1, x2, x3 = symbols('x1 x2 x3', cls=Dummy)
                res=transformation(*sbs)
                transformation = Lambda(sbs,res)
                n_t0=len(sbs)
                n_t1=len(res)
            elif isinstance(transformation, string_types):
                transformation = Symbol(transformation)
            elif isinstance(transformation, (Symbol, Lambda)):
                pass
            else:
                raise TypeError("transformation: "
                                "wrong type {0}".format(type(transformation)))

        # figure out the dimension n and check that there are no contradictions
        n_dim = dim 
        n_p=None if parent is None else parent.dim
        n_l=None
        if hasattr(location,'dim'):
            n_l=location.dim
        n_r0=None if rotation_matrix is None else rotation_matrix.shape[0]
        n_r1=None if rotation_matrix is None else rotation_matrix.shape[1]
        n_vec=None if vector_names is None else len(vector_names)
        n_var=None if variable_names is None else len(variable_names)

        dims=[n for n in [n_dim,n_p,n_l,n_r0,n_r1,n_t0,n_t1,n_vec,n_var] if n is not None]
        assert len(dims)>0  #make sure we have some information about the dimension
        assert any([dims[0]==d for d in dims]) # and it is consistend
        n=dims[0]

        # If orientation information has been provided, store
        # the rotation matrix accordingly
        if rotation_matrix is None:
            rotation_matrix = ImmutableDenseMatrix(eye(n))
        else:
            if not isinstance(rotation_matrix, MatrixBase):
                raise TypeError("rotation_matrix should be an Immutable" +
                                "Matrix instance")
            rotation_matrix = rotation_matrix.as_immutable()

        # If location information is not given, adjust the default
        # location as Vector.zero
        if parent is not None:
            if not isinstance(parent, CoordSysND):
                raise TypeError("parent should be a " +
                                "CoordSysND/None")
            if location is None:
                location = Vector.zero
            else:
                if not isinstance(location, Vector):
                    raise TypeError("location should be a Vector")
                # Check that location does not contain base
                # scalars
                for x in location.free_symbols:
                    if isinstance(x, BaseScalar):
                        raise ValueError("location should not contain" +
                                         " BaseScalars")
            origin = parent.origin.locate_new(name + '.origin',
                                              location)
        else:
            location = Vector.zero
            origin = Point(name + '.origin')

        if transformation is None:
            transformation = Tuple(rotation_matrix, location)

        if isinstance(transformation, Tuple):
            lambda_transformation = CoordSysND._compose_rotation_and_translation(
                transformation[0],
                transformation[1],
                parent
            )
            r, l = transformation
            l = l._projections(n)
            lambda_lame = CoordSysND._get_lame_coeff('cartesian',n)
            #lambda_inverse = lambda x, y, z: r.inv()*Matrix([x-l[0], y-l[1], z-l[2]])
            #sbs=symbols(["x{0}".format(i) for i in range(n)])
            sbs=cls.dummy_symbols(n)
            lambda_inverse = lambda *sbs: r.inv()*Matrix([sbs[i]-l[i] for i in range(n)])

        elif isinstance(transformation, Symbol):
            trname = transformation.name
            lambda_transformation = CoordSysND._get_transformation_lambdas(trname)
            if parent is not None:
                if parent.lame_coefficients() != tuple([S(1) for i in range(n)]):
                    raise ValueError('Parent for pre-defined coordinate '
                                 'system should be Cartesian.')
            lambda_lame = CoordSysND._get_lame_coeff(trname,n)
            lambda_inverse = CoordSysND._set_inv_trans_equations(trname)
        elif isinstance(transformation, Lambda):
            if not CoordSysND._check_orthogonality(transformation,n):
                raise ValueError("The transformation equation does not "
                                 "create orthogonal coordinate system")
            lambda_transformation = transformation
            lambda_lame = cls._calculate_lame_coeff(lambda_transformation,n)
            lambda_inverse = None
        else:
            sbs=symbols(["x{0}".format(i) for i in range(n)])
            lambda_transformation = lambda *sbs: transformation(*sbs)
            lambda_lame = CoordSysND._get_lame_coeff(transformation,n)
            lambda_inverse = None

        if variable_names is None:
            if isinstance(transformation, Lambda):
                #variable_names = ["x1", "x2", "x3"]
                pass
                source_tup=transformation.args[0]
                variable_names= [str(var) for var in source_tup] 
            elif isinstance(transformation, Symbol):
                if transformation.name is 'hyperspherical':
                    variable_names = ["r"] + ["phi_{0}".format(i) for i in range(1,n)]
                elif transformation.name is 'cartesian':
                    variable_names = ["x_{0}".format(i) for i in range(0,n)]
                #elif transformation.name is 'cylindrical':
                #    variable_names = ["r", "theta", "z"]
                #else:
                #    variable_names = ["x","y","z"]
            else:
                variable_names = ["x_{0}".format(i) for i in range(n)]
        if vector_names is None:
            vector_names = ["e_{0}".format(i) for i in range(n)]

        # All systems that are defined as 'roots' are unequal, unless
        # they have the same name.
        # Systems defined at same orientation/position wrt the same
        # 'parent' are equal, irrespective of the name.
        # This is true even if the same orientation is provided via
        # different methods like Axis/Body/Space/Quaternion.
        # However, coincident systems may be seen as unequal if
        # positioned/oriented wrt different parents, even though
        # they may actually be 'coincident' wrt the root system.
        if parent is not None:
            obj = super(CoordSysND, cls).__new__(
                cls, Symbol(name), transformation, parent)
        else:
            obj = super(CoordSysND, cls).__new__(
                cls, Symbol(name), transformation)

        obj._name = name
        obj.dim=n
        # Initialize the base vectors

        _check_strings('vector_names', vector_names,obj.dim)
        vector_names = list(vector_names)
        latex_vects = [(r'\mathbf{\hat{%s}_{%s}}' % (x, name)) for
                           x in vector_names]
        pretty_vects = [(name + '_' + x) for x in vector_names]

        obj._vector_names = vector_names
        base_vects =[ BaseVector(i, obj, pretty_vects[i], latex_vects[i]) for i in range(n) ]

        obj._base_vectors = tuple(base_vects)

        # Initialize the base scalars

        _check_strings('variable_names', vector_names,obj.dim)
        variable_names = list(variable_names)
        latex_scalars = [(r"\mathbf{{%s}_{%s}}" % (x, name)) for
                         x in variable_names]
        pretty_scalars = [(name + '_' + x) for x in variable_names]

        obj._variable_names = variable_names
        obj._vector_names = vector_names

        base_scalars=[ BaseScalar(i, obj, pretty_scalars[i], latex_scalars[i]) for i in range(n) ]

        obj._base_scalars = base_scalars

        obj._transformation = transformation
        obj._transformation_lambda = lambda_transformation
        #obj._lame_coefficients = lambda_lame(x1, x2, x3)
    
    
        obj._lame_coefficients = lambda_lame(*base_scalars)
        obj._transformation_from_parent_lambda = lambda_inverse
        
        for i in range(n):
            setattr(obj, variable_names[i], base_scalars[i])
            setattr(obj, vector_names[i], base_vects[i])

        # Assign params
        obj._parent = parent
        if obj._parent is not None:
            obj._root = obj._parent._root
        else:
            obj._root = obj

        obj._parent_rotation_matrix = rotation_matrix
        obj._origin = origin

        # Return the instance
        return obj

    def __str__(self, printer=None):
        return self._name

    __repr__ = __str__
    _sympystr = __str__

    def __iter__(self):
        return iter(self.base_vectors())

    @staticmethod
    def _check_orthogonality(equations,dim):
        """
        Helper method for _connect_to_cartesian. It checks if
        set of transformation equations create orthogonal curvilinear
        coordinate system

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations

        """


        sbs=symbols(["x{0}".format(i) for i in range(dim)])
        #equations = equations(x1, x2, x3)
        #v1 = Matrix([diff(equations[0], x1),
        #             diff(equations[1], x1), diff(equations[2], x1)])

        #v2 = Matrix([diff(equations[0], x2),
        #             diff(equations[1], x2), diff(equations[2], x2)])

        #v3 = Matrix([diff(equations[0], x3),
        #             diff(equations[1], x3), diff(equations[2], x3)])
        equations = equations(*sbs)
        vs=[ Matrix([diff(equations[i], sbs[j]) for i in range(dim)]) for j in range(dim )]
        
        #if any(simplify(i[0] + i[1] + i[2]) == 0 for i in (v1, v2, v3)):
        if any([simplify(sum([v[i] for i in range(dim)])) == 0 for v in vs]):
            return False
        else:
            #if simplify(v1.dot(v2)) == 0 and simplify(v2.dot(v3)) == 0 \
            #    and simplify(v3.dot(v1)) == 0:
            if all([simplify(vs[i].dot(vs[j]))==0 for i in range(dim) for j in range(i+1,dim)]): 
                return True
            else:
                return False

    @staticmethod
    def _set_inv_trans_equations(curv_coord_name):
        """
        Store information about inverse transformation equations for
        pre-defined coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if curv_coord_name == 'cartesian':
            return lambda x, y, z: (x, y, z)

        if curv_coord_name == 'spherical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2 + z**2),
                acos(z/sqrt(x**2 + y**2 + z**2)),
                atan2(y, x)
            )
        if curv_coord_name == 'cylindrical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2),
                atan2(y, x),
                z
            )
        raise ValueError('Wrong set of parameters.'
                         'Type of coordinate system is defined')

    def _calculate_inv_trans_equations(self):
        """
        Helper method for set_coordinate_type. It calculates inverse
        transformation equations for given transformations equations.

        """
        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy, reals=True)
        x, y, z = symbols("x, y, z", cls=Dummy)

        equations = self._transformation(x1, x2, x3)

        try:
            solved = solve([equations[0] - x,
                            equations[1] - y,
                            equations[2] - z], (x1, x2, x3), dict=True)[0]
            solved = solved[x1], solved[x2], solved[x3]
            self._transformation_from_parent_lambda = \
                lambda x1, x2, x3: tuple(i.subs(list(zip((x, y, z), (x1, x2, x3)))) for i in solved)
        except:
            raise ValueError('Wrong set of parameters.')

    @staticmethod
    def _get_lame_coeff(curv_coord_name,dim=None):
        """
        Store information about Lame coefficients for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, string_types):
            if curv_coord_name == 'cartesian':
                #return lambda x, y, z: (S.One, S.One, S.One)
                sbs=symbols(["x{0}".format(i) for i in range(dim)])
                return lambda *sbs: tuple([S.One for i in range(dim)])
            #if curv_coord_name == 'spherical':
            #    return lambda r, theta, phi: (S.One, r, r*sin(theta))
            #if curv_coord_name == 'cylindrical':
            #    return lambda r, theta, h: (S.One, r, S.One)
            raise ValueError('Wrong set of parameters.'
                             ' Type of coordinate system is not defined')
        #return CoordSysND._calculate_lame_coefficients(curv_coord_name)
    
    @classmethod
    def dummy_symbols(cls,dim):
        return symbols(["x{0}".format(i) for i in range(dim)])

    @classmethod
    def _calculate_lame_coeff(cls,equations,dim):
        """
        It calculates Lame coefficients
        for given transformations equations.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations.

        """
        #return lambda x1, x2, x3: (
        #                  sqrt(diff(equations(x1, x2, x3)[0], x1)**2 +
        #                       diff(equations(x1, x2, x3)[1], x1)**2 +
        #                       diff(equations(x1, x2, x3)[2], x1)**2),
        #                  sqrt(diff(equations(x1, x2, x3)[0], x2)**2 +
        #                       diff(equations(x1, x2, x3)[1], x2)**2 +
        #                       diff(equations(x1, x2, x3)[2], x2)**2),
        #                  sqrt(diff(equations(x1, x2, x3)[0], x3)**2 +
        #                       diff(equations(x1, x2, x3)[1], x3)**2 +
        #                       diff(equations(x1, x2, x3)[2], x3)**2)
        #              )
        sbs=cls.dummy_symbols(dim)
        res=tuple(
            [   
                sqrt(
                    sum(
                        [
                            diff(equations(*sbs)[i],sbs[j])**2 for i in range(dim)
                        ]
                    )
                )
                for j in range(dim)
            ]
        )
        return lambda *sbs:res

    def _inverse_rotation_matrix(self):
        """
        Returns inverse rotation matrix.
        """
        return simplify(self._parent_rotation_matrix**-1)

    @staticmethod
    def _get_transformation_lambdas(curv_coord_name):
        """
        Store information about transformation equations for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, string_types):
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (x, y, z)
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (
                    r*sin(theta)*cos(phi),
                    r*sin(theta)*sin(phi),
                    r*cos(theta)
                )
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (
                    r*cos(theta),
                    r*sin(theta),
                    h
                )
            raise ValueError('Wrong set of parameters.'
                             'Type of coordinate system is defined')

    @classmethod
    def _rotation_trans_equations(cls, matrix, equations):
        """
        Returns the transformation equations obtained from rotation matrix.

        Parameters
        ==========

        matrix : Matrix
            Rotation matrix

        equations : tuple
            Transformation equations

        """
        return tuple(matrix * Matrix(equations))

    @property
    def origin(self):
        return self._origin

    @property
    def delop(self):
        SymPyDeprecationWarning(
            feature="coord_system.delop has been replaced.",
            useinstead="Use the Del() class",
            deprecated_since_version="1.1",
            issue=12866,
        ).warn()
        from sympy.vector.deloperator import Del
        return Del()

    def base_vectors(self):
        return self._base_vectors

    def base_scalars(self):
        return self._base_scalars

    def lame_coefficients(self):
        return self._lame_coefficients

    def transformation_to_parent(self):
        return self._transformation_lambda(*self.base_scalars())

    def transformation_from_parent(self):
        if self._parent is None:
            raise ValueError("no parent coordinate system, use "
                             "`transformation_from_parent_function()`")
        return self._transformation_from_parent_lambda(
                            *self._parent.base_scalars())

    def transformation_from_parent_function(self):
        return self._transformation_from_parent_lambda

    def rotation_matrix(self, other):
        """
        Returns the direction cosine matrix(DCM), also known as the
        'rotation matrix' of this coordinate system with respect to
        another system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.rotation_matrix(B) * v_b.

        A SymPy Matrix is returned.

        Parameters
        ==========

        other : CoordSysND
            The system which the DCM is generated to.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSysND('N')
        >>> A = N.orient_new_axis('A', q1, N.i)
        >>> N.rotation_matrix(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        """
        from sympy.vector.nd_functions import _path
        if not isinstance(other, CoordSysND):
            raise TypeError(str(other) +
                            " is not a CoordSysND")
        # Handle special cases
        n=self.dim
        if other == self:
            return eye(n)
        elif other == self._parent:
            return self._parent_rotation_matrix
        elif other._parent == self:
            return other._parent_rotation_matrix.T
        # Else, use tree to calculate position
        rootindex, path = _path(self, other)
        result = eye(n)
        i = -1
        for i in range(rootindex):
            result *= path[i]._parent_rotation_matrix
        i += 2
        while i < len(path):
            result *= path[i]._parent_rotation_matrix.T
            i += 1
        return result

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of the origin of this coordinate
        system with respect to another Point/CoordSysND.

        Parameters
        ==========

        other : Point/CoordSysND
            If other is a Point, the position of this system's origin
            wrt it is returned. If its an instance of CoordSyRect,
            the position wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> N = CoordSysND('N')
        >>> N1 = N.locate_new('N1', 10 * N.i)
        >>> N.position_wrt(N1)
        (-10)*N.i

        """
        return self.origin.position_wrt(other)

    def scalar_map(self, other):
        """
        Returns a dictionary which expresses the coordinate variables
        (base scalars) of this frame in terms of the variables of
        otherframe.

        Parameters
        ==========

        otherframe : CoordSysND
            The other system to map the variables to.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> from sympy import Symbol
        >>> A = CoordSysND('A')
        >>> q = Symbol('q')
        >>> B = A.orient_new_axis('B', q, A.k)
        >>> A.scalar_map(B)
        {A.x: B.x*cos(q) - B.y*sin(q), A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}

        """

        relocated_scalars = []
        origin_coords = tuple(self.position_wrt(other).to_matrix(other))
        for i, x in enumerate(other.base_scalars()):
            relocated_scalars.append(x - origin_coords[i])

        vars_matrix = (self.rotation_matrix(other) *
                       Matrix(relocated_scalars))
        mapping = {}
        for i, x in enumerate(self.base_scalars()):
            mapping[x] = trigsimp(vars_matrix[i])
        return mapping

    def locate_new(self, name, position, vector_names=None,
                   variable_names=None):
        """
        Returns a CoordSysND with its origin located at the given
        position wrt this coordinate system's origin.

        Parameters
        ==========

        name : str
            The name of the new CoordSysND instance.

        position : Vector
            The position vector of the new system's origin wrt this
            one.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> A = CoordSysND('A')
        >>> B = A.locate_new('B', 10 * A.i)
        >>> B.origin.position_wrt(A.origin)
        10*A.i

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        return CoordSysND(name, location=position,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          parent=self)

    def orient_new(self, name, orienters, location=None,
                   vector_names=None, variable_names=None):
        """
        Creates a new CoordSysND oriented in the user-specified way
        with respect to this system.

        Please refer to the documentation of the orienter classes
        for more information about the orientation procedure.

        Parameters
        ==========

        name : str
            The name of the new CoordSysND instance.

        orienters : iterable/Orienter
            An Orienter or an iterable of Orienters for orienting the
            new coordinate system.
            If an Orienter is provided, it is applied to get the new
            system.
            If an iterable is provided, the orienters will be applied
            in the order in which they appear in the iterable.

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSysND('N')

        Using an AxisOrienter

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> A = N.orient_new('A', (axis_orienter, ))

        Using a BodyOrienter

        >>> from sympy.vector import BodyOrienter
        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> B = N.orient_new('B', (body_orienter, ))

        Using a SpaceOrienter

        >>> from sympy.vector import SpaceOrienter
        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> C = N.orient_new('C', (space_orienter, ))

        Using a QuaternionOrienter

        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> D = N.orient_new('D', (q_orienter, ))
        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        if isinstance(orienters, Orienter):
            if isinstance(orienters, AxisOrienter):
                final_matrix = orienters.rotation_matrix(self)
            else:
                final_matrix = orienters.rotation_matrix()
            # TODO: trigsimp is needed here so that the matrix becomes
            # canonical (scalar_map also calls trigsimp; without this, you can
            # end up with the same CoordinateSystem that compares differently
            # due to a differently formatted matrix). However, this is
            # probably not so good for performance.
            final_matrix = trigsimp(final_matrix)
        else:
            final_matrix = Matrix(eye(3))
            for orienter in orienters:
                if isinstance(orienter, AxisOrienter):
                    final_matrix *= orienter.rotation_matrix(self)
                else:
                    final_matrix *= orienter.rotation_matrix()

        return CoordSysND(name, rotation_matrix=final_matrix,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          location=location,
                          parent=self)

    def orient_new_axis(self, name, angle, axis, location=None,
                        vector_names=None, variable_names=None):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSysND('N')
        >>> B = N.orient_new_axis('B', q1, N.i + 2 * N.j)

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        orienter = AxisOrienter(angle, axis)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_body(self, name, angle1, angle2, angle3,
                        rotation_order, location=None,
                        vector_names=None, variable_names=None):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSysND('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> D = N.orient_new_body('D', q1, q2, q3, '123')

        is same as

        >>> D = N.orient_new_axis('D', q1, N.i)
        >>> D = D.orient_new_axis('D', q2, D.j)
        >>> D = D.orient_new_axis('D', q3, D.k)

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> B = N.orient_new_body('B', q1, q2, q3, '123')
        >>> B = N.orient_new_body('B', q1, q2, 0, 'ZXZ')
        >>> B = N.orient_new_body('B', 0, 0, 0, 'XYX')

        """

        orienter = BodyOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_space(self, name, angle1, angle2, angle3,
                         rotation_order, location=None,
                         vector_names=None, variable_names=None):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        See Also
        ========

        CoordSysND.orient_new_body : method to orient via Euler
            angles

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSysND('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> D = N.orient_new_space('D', q1, q2, q3, '312')

        is same as

        >>> B = N.orient_new_axis('B', q1, N.i)
        >>> C = B.orient_new_axis('C', q2, N.j)
        >>> D = C.orient_new_axis('D', q3, N.k)

        """

        orienter = SpaceOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_quaternion(self, name, q0, q1, q2, q3, location=None,
                              vector_names=None, variable_names=None):
        """
        Quaternion orientation orients the new CoordSysND with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSysND('N')
        >>> B = N.orient_new_quaternion('B', q0, q1, q2, q3)

        """

        orienter = QuaternionOrienter(q0, q1, q2, q3)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def create_new(self, name, transformation, variable_names=None, vector_names=None):
        """
        Returns a CoordSysND which is connected to self by transformation.

        Parameters
        ==========

        name : str
            The name of the new CoordSysND instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSysND
        >>> a = CoordSysND('a')
        >>> b = a.create_new('b', transformation='spherical')
        >>> b.transformation_to_parent()
        (b.r*sin(b.theta)*cos(b.phi), b.r*sin(b.phi)*sin(b.theta), b.r*cos(b.theta))
        >>> b.transformation_from_parent()
        (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))

        """
        return CoordSysND(name, parent=self, transformation=transformation,
                          variable_names=variable_names, vector_names=vector_names)

    def __init__(self, name, location=None, rotation_matrix=None,
                 parent=None, vector_names=None, variable_names=None,
                 latex_vects=None, pretty_vects=None, latex_scalars=None,
                 pretty_scalars=None, transformation=None):
        # Dummy initializer for setting docstring
        pass

    __init__.__doc__ = __new__.__doc__

    @staticmethod
    def _compose_rotation_and_translation(rot, translation, parent):
        n=rot.shape[0] 
        sbs=symbols(["x{0}".format(i) for i in range(n)])

        #r = lambda x, y, z: CoordSysND._rotation_trans_equations(rot, (x, y, z))
        r = lambda *sbs: CoordSysND._rotation_trans_equations(rot, tuple(sbs))

        if parent is None:
            return r

        #dx, dy, dz = [translation.dot(i) for i in parent.base_vectors()]
        #t = lambda x, y, z: (
        #    x + dx,
        #    y + dy,
        #    z + dz,
        #)
        #return lambda x, y, z: t(*r(x, y, z))
        dsbs = [translation.dot(i) for i in parent.base_vectors()]
        t = lambda *sbs: tuple([sbs[i] +dsbs[i] for i in range(n) ])

        return lambda *sbs: t(*r(*sbs))


def _check_strings(arg_name, arg, dim):
    errorstr = arg_name + " must be an iterable of "+str(dim)+" string-types"
    if len(arg) != dim:
        raise ValueError(errorstr)
    for s in arg:
        if not isinstance(s, string_types):
            raise TypeError(errorstr)
