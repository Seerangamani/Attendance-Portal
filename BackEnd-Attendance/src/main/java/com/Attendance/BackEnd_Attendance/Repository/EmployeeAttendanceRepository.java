package com.Attendance.BackEnd_Attendance.Repository;

import com.Attendance.BackEnd_Attendance.Model.Attendance;
import com.Attendance.BackEnd_Attendance.Model.EmployeeAttendanceDTO;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.jpa.repository.Query;
import java.util.List;

public interface EmployeeAttendanceRepository extends CrudRepository<Attendance, String> {

    @Query("SELECT new com.Attendance.BackEnd_Attendance.Model.EmployeeAttendanceDTO(" +
            "a.employeeId, a.employeeName, a.department, a.date, a.checkinTime, a.checkoutTime, a.status) " +
            "FROM Attendance a WHERE a.date = CURRENT_DATE")
    List<EmployeeAttendanceDTO> findTodayEmployeeAttendance();

    @Query("SELECT new com.Attendance.BackEnd_Attendance.Model.EmployeeAttendanceDTO(" +
            "a.employeeId, a.employeeName, a.department, a.date, a.checkinTime, a.checkoutTime, a.status) " +
            "FROM Attendance a")
    List<EmployeeAttendanceDTO> findAllEmployeeAttendance();
}
