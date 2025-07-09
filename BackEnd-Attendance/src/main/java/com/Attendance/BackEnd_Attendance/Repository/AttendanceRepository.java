package com.Attendance.BackEnd_Attendance.Repository;

import com.Attendance.BackEnd_Attendance.Model.Attendance;
import com.Attendance.BackEnd_Attendance.Model.Employee;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public interface AttendanceRepository extends JpaRepository<Attendance, Long> {

    // Find all attendance records for an employee
    List<Attendance> findByEmployeeId(String employeeId);

    // Find attendance records for a specific employee and date
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date")
    List<Attendance> findByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // Find single attendance record for employee and date
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date")
    Optional<Attendance> findAttendanceByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // Find attendance records between dates for an employee
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date BETWEEN :startDate AND :endDate ORDER BY a.date DESC")
    List<Attendance> findByEmployeeIdAndDateBetween(@Param("employeeId") String employeeId, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);

    // Find all attendance records for a specific date
    @Query("SELECT a FROM Attendance a WHERE a.date = :date ORDER BY a.employeeId")
    List<Attendance> findByDate(@Param("date") LocalDate date);

    // Find attendance records between dates
    @Query("SELECT a FROM Attendance a WHERE a.date BETWEEN :startDate AND :endDate ORDER BY a.date DESC, a.employeeId")
    List<Attendance> findByDateBetween(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);

    // Check if attendance exists for employee and date
    @Query("SELECT CASE WHEN COUNT(a) > 0 THEN true ELSE false END FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date")
    boolean existsByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // Find active attendance (with checkin or status) for employee and date
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date AND (a.checkinTime IS NOT NULL OR a.status IS NOT NULL)")
    Optional<Attendance> findActiveAttendanceByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // Count attendance records for employee and date
    @Query("SELECT COUNT(a) FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date")
    long countByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // Find latest attendance record for an employee
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId ORDER BY a.date DESC, a.id DESC LIMIT 1")
    List<Attendance> findLatestByEmployeeId(@Param("employeeId") String employeeId);

    // Find attendance records for today with check-in time
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date AND a.checkinTime IS NOT NULL")
    List<Attendance> findByEmployeeIdAndDateWithCheckin(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // Debug query to find all records for an employee
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId ORDER BY a.date DESC")
    List<Attendance> findAllByEmployeeIdOrderByDateDesc(@Param("employeeId") String employeeId);

    // Find attendance records that are eligible for checkout
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date AND a.checkinTime IS NOT NULL AND a.checkoutTime IS NULL AND a.status NOT IN ('Absent', 'Permission Granted')")
    List<Attendance> findEligibleForCheckout(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    List<Attendance> findAllByEmployeeId(String employeeId);

    @Query("SELECT a FROM Attendance a WHERE a.employeeId = ?1 AND a.date = ?2 AND " +
            "(a.status = 'Permission' OR a.status = 'Half Day')")
    Optional<Attendance> findAllocationByEmployeeIdAndDate(String employeeId, LocalDate date);

    List<Attendance> findTop1ByEmployeeIdOrderByDateDesc(String employeeId);

    List<Attendance> findByEmployeeIdAndDateBetweenOrderByDateAsc(String employeeId, LocalDate startDate, LocalDate endDate);

    // Department-wise attendance summary - FIXED QUERY
    @Query(value = "SELECT e.department, COUNT(a.employee_id) as present_count " +
            "FROM attendance a " +
            "JOIN employee e ON a.employee_id = e.id " +
            "WHERE a.date = :today AND a.status = 'Present' " +
            "GROUP BY e.department",
            nativeQuery = true)
    List<Object[]> getDepartmentWiseAttendance(@Param("today") LocalDate today);

    // Monthly attendance summary - customize this query based on your requirements
    @Query("SELECT a.employeeId, COUNT(a.id) as attendance_count, " +
            "SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as present_days, " +
            "SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) as absent_days " +
            "FROM Attendance a " +
            "WHERE a.date BETWEEN :monthStart AND :today " +
            "GROUP BY a.employeeId")
    List<Object[]> getMonthlyAttendanceSummary(@Param("monthStart") LocalDate monthStart, @Param("today") LocalDate today);

    // ✅ FIXED: Find attendance records by employee ID, date, and status (returns List instead of single record)
    List<Attendance> findByEmployeeIdAndDateAndStatus(String employeeId, LocalDate date, String status);

    // ✅ NEW: Find SINGLE attendance record by employee ID, date, and status (for permission/halfday checks)
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date AND a.status = :status")
    Optional<Attendance> findSingleByEmployeeIdAndDateAndStatus(@Param("employeeId") String employeeId,
                                                                @Param("date") LocalDate date,
                                                                @Param("status") String status);

    // ✅ NEW: Find attendance record with specific status that returns single result (for your service)
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date AND a.status = :status ORDER BY a.id DESC")
    Optional<Attendance> findFirstByEmployeeIdAndDateAndStatusOrderByIdDesc(@Param("employeeId") String employeeId,
                                                                            @Param("date") LocalDate date,
                                                                            @Param("status") String status);

    /**
     * Find attendance records by employee ID and date between range with specific status
     */
    List<Attendance> findByEmployeeIdAndDateBetweenAndStatus(String employeeId, LocalDate startDate, LocalDate endDate, String status);

    // ✅ NEW: Check if permission is allocated for employee on specific date
    @Query("SELECT CASE WHEN COUNT(a) > 0 THEN true ELSE false END FROM Attendance a " +
            "WHERE a.employeeId = :employeeId AND a.date = :date AND a.status = 'PERMISSION' " +
            "AND a.permissionFrom IS NOT NULL AND a.permissionTo IS NOT NULL")
    boolean hasPermissionAllocated(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // ✅ NEW: Check if half day is allocated for employee on specific date
    @Query("SELECT CASE WHEN COUNT(a) > 0 THEN true ELSE false END FROM Attendance a " +
            "WHERE a.employeeId = :employeeId AND a.date = :date AND a.status = 'HALF_DAY' " +
            "AND a.halfdayFrom IS NOT NULL AND a.halfdayTo IS NOT NULL")
    boolean hasHalfDayAllocated(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // ✅ NEW: Find permission allocation for employee and date
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date " +
            "AND a.status = 'PERMISSION' AND a.permissionFrom IS NOT NULL AND a.permissionTo IS NOT NULL")
    Optional<Attendance> findPermissionAllocation(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // ✅ NEW: Find half day allocation for employee and date
    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date " +
            "AND a.status = 'HALF_DAY' AND a.halfdayFrom IS NOT NULL AND a.halfdayTo IS NOT NULL")
    Optional<Attendance> findHalfDayAllocation(@Param("employeeId") String employeeId, @Param("date") LocalDate date);

    // ✅ NEW: Find all permission/halfday allocations for a date (for admin dashboard)
    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.status IN ('PERMISSION', 'HALF_DAY') " +
            "ORDER BY a.employeeId")
    List<Attendance> findAllPermissionHalfdayAllocations(@Param("date") LocalDate date);

    // ✅ NEW: Find employees who haven't used their permission/halfday yet
    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.status IN ('PERMISSION', 'HALF_DAY') " +
            "AND a.checkinTime IS NULL ORDER BY a.employeeId")
    List<Attendance> findUnusedPermissionHalfdayAllocations(@Param("date") LocalDate date);

    // ✅ NEW: Find employees who came late for permission/halfday
    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.status IN ('LATE', 'ABSENT') " +
            "AND (a.permissionFrom IS NOT NULL OR a.halfdayFrom IS NOT NULL) ORDER BY a.employeeId")
    List<Attendance> findLatePermissionHalfdayUsers(@Param("date") LocalDate date);

    List<Attendance> findByDateAndStatus(LocalDate date, String holiday);

    @Query("SELECT e FROM Employee e")
    List<Employee> findAllEmployees();
}