program dump_bmad_attributes
    ! Dump every element attribute known to Bmad as a pipe-delimited table:
    !   ELEMENT|ATTR|STATE|KIND|UNITS
    ! Names are upper-cased. Descriptions are not available here (they live in
    ! the reference manual); gen_attrs.py merges them in from elements.tex.
    use bmad
    implicit none

    integer :: i_key, i_attrib
    type (ele_struct) :: ele
    type (ele_attribute_struct) :: info

    character(60) :: key_str

    do i_key = 1, n_key$
        ele%key = i_key
        key_str = key_name(i_key)

        ! Skip invalid keys
        if (trim(key_str) == "" .or. trim(key_str) == "!!!") cycle

        do i_attrib = 1, num_ele_attrib_extended$
            info = attribute_info(ele, i_attrib)

            if (info%name(1:1) == '!') cycle
            if (info%state == does_not_exist$) cycle

            write(*, '(9A)') &
                trim(upcase(trim(key_str))), '|', &
                trim(upcase(trim(info%name))), '|', &
                trim(get_state_enum(info%state)), '|', &
                trim(get_kind_enum(info%kind)), '|', &
                trim(info%units)
        end do
    end do

contains

    function get_state_enum(state_int) result(s_str)
        integer, intent(in) :: state_int
        character(30) :: s_str

        select case (state_int)
        case (is_free$)
            s_str = "Free"
        case (quasi_free$)
            s_str = "Quasi_Free"
        case (dependent$)
            s_str = "Dependent"
        case (private$)
            s_str = "Private"
        case (overlay_slave$)
            s_str = "Overlay_Slave"
        case (field_master_dependent$)
            s_str = "Field_Master_Dependent"
        case (super_lord_align$)
            s_str = "Super_Lord_Align"
        case default
            s_str = "Unknown"
        end select
    end function get_state_enum

    function get_kind_enum(kind_int) result(k_str)
        integer, intent(in) :: kind_int
        character(30) :: k_str

        select case (kind_int)
        case (is_real$)
            k_str = "Real"
        case (is_integer$)
            k_str = "Integer"
        case (is_logical$)
            k_str = "Logical"
        case (is_switch$)
            k_str = "Switch"
        case (is_string$)
            k_str = "String"
        case (is_struct$)
            k_str = "Struct"
        case default
            k_str = "Unknown"
        end select
    end function get_kind_enum

end program dump_bmad_attributes
