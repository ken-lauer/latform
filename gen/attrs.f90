program generate_python_attributes
    use bmad
    implicit none

    integer :: i_key, i_attrib
    type (ele_struct) :: ele
    type (ele_attribute_struct) :: info

    character(60) :: key_str
    character(60) :: state_str
    character(60) :: kind_str
    character(200) :: unit_str
    logical :: first_attrib, element_has_content

    write(*, "(A)") 'from dataclasses import dataclass'
    write(*, "(A)") "from enum import Enum"
    write(*, "(A)") ""
    write(*, "(A)") "class State(str, Enum):"
    write(*, "(A)") "    Does_Not_Exist = ""Does_Not_Exist"""
    write(*, "(A)") "    Free = ""Free"""
    write(*, "(A)") "    Quasi_Free = ""Quasi_Free"""
    write(*, "(A)") "    Dependent = ""Dependent"""
    write(*, "(A)") "    Private = ""Private"""
    write(*, "(A)") "    Overlay_Slave = ""Overlay_Slave"""
    write(*, "(A)") "    Field_Master_Dependent = ""Field_Master_Dependent"""
    write(*, "(A)") "    Super_Lord_Align = ""Super_Lord_Align"""
    write(*, "(A)") "    Unknown = ""Unknown"""
    write(*, "(A)") ""
    write(*, "(A)") "class Kind(Enum):"
    write(*, "(A)") "    Real = ""Real"""
    write(*, "(A)") "    Integer = ""Integer"""
    write(*, "(A)") "    Logical = ""Logical"""
    write(*, "(A)") "    Switch = ""Switch"""
    write(*, "(A)") "    String = ""String"""
    write(*, "(A)") "    Struct = ""Struct"""
    write(*, "(A)") "    Unknown = ""Unknown"""
    write(*, "(A)") ""
    write(*, "(A)") "@dataclass"
    write(*, "(A)") "class Attr:"
    write(*, "(A)") "    name: str"
    write(*, "(A)") "    state: State"
    write(*, "(A)") "    kind: Kind"
    write(*, "(A)") "    units: str"
    write(*, "(A)") ""
    write(*, "(A)") "by_element: dict[str, dict[str, Attr]] = {}"
    write(*, "(A)") ""

    do i_key = 1, n_key$
        ele%key = i_key
        key_str = key_name(i_key)

        ! Skip invalid keys
        if (trim(key_str) == "" .or. trim(key_str) == "!!!") cycle

        element_has_content = .false.
        do i_attrib = 1, num_ele_attrib_extended$
            info = attribute_info(ele, i_attrib)
            if (info%name(1:1) == '!') cycle
            if (info%state == does_not_exist$) cycle
            element_has_content = .true.
            exit
        end do

        if (.not. element_has_content) cycle

        write(*, '(A, A, A)') 'by_element["', upcase(trim(key_str)), '"] = {'

        first_attrib = .true.

        do i_attrib = 1, num_ele_attrib_extended$

            info = attribute_info(ele, i_attrib)

            if (info%name(1:1) == '!') cycle
            if (info%state == does_not_exist$) cycle

            state_str = get_state_enum(info%state)
            kind_str  = get_kind_enum(info%kind)

            unit_str = trim(info%units)
            if (trim(unit_str) == "") unit_str = ""

            if (.not. first_attrib) then
                 write(*, '(A)') ","
            endif
            first_attrib = .false.

            ! "L": Attr("L", State.Free, Kind.Real, units="m")
            write(*, '(20A)', advance='no') &
                '    "', upcase(trim(info%name)), '": Attr(', &
                '"', upcase(trim(info%name)), '", ', &
                trim(state_str), ', ', &
                trim(kind_str), ', ', &
                'units="', trim(unit_str), '")'

        end do

        write(*, '(A)') ""    ! Finish last line
        write(*, '(A)') "}"   ! Close brace
        write(*, '(A)') ""    ! Empty line between blocks

    end do

contains

    function get_state_enum(state_int) result(s_str)
        integer, intent(in) :: state_int
        character(60) :: s_str

        select case (state_int)
        case (is_free$)
            s_str = "State.Free"
        case (quasi_free$)
            s_str = "State.Quasi_Free"
        case (dependent$)
            s_str = "State.Dependent"
        case (private$)
            s_str = "State.Private"
        case (overlay_slave$)
            s_str = "State.Overlay_Slave"
        case (field_master_dependent$)
            s_str = "State.Field_Master_Dependent"
        case (super_lord_align$)
            s_str = "State.Super_Lord_Align"
        case default
            s_str = "State.Unknown"
        end select
    end function get_state_enum

    function get_kind_enum(kind_int) result(k_str)
        integer, intent(in) :: kind_int
        character(60) :: k_str

        select case (kind_int)
        case (is_real$)
            k_str = "Kind.Real"
        case (is_integer$)
            k_str = "Kind.Integer"
        case (is_logical$)
            k_str = "Kind.Logical"
        case (is_switch$)
            k_str = "Kind.Switch"
        case (is_string$)
            k_str = "Kind.String"
        case (is_struct$)
            k_str = "Kind.Struct"
        case default
            k_str = "Kind.Unknown"
        end select
    end function get_kind_enum

end program generate_python_attributes
