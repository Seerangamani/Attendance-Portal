//package com.Attendance.BackEnd_Attendance.Repository;
//
//import com.Attendance.BackEnd_Attendance.Model.Attendance;
//import org.springframework.data.jpa.repository.JpaRepository;
//import org.springframework.data.jpa.repository.Query;
//import org.springframework.data.repository.query.Param;
//import org.springframework.stereotype.Repository;
//
//import java.time.LocalDate;
//import java.time.LocalTime;
//import java.util.List;
//
//@Repository
//public interface AttendanceConditionRepository extends JpaRepository<Attendance, Long> {
//
//    // Basic queries by employee ID
//    List<Attendance> findByEmployeeId(String employeeId);
//
//    List<Attendance> findByEmployeeIdOrderByDateDesc(String employeeId);
//
//    List<Attendance> findByEmployeeIdOrderByDateAsc(String employeeId);
//
//    // Date-based queries
//    Attendance findByEmployeeIdAndDate(String employeeId, LocalDate date);
//
//    List<Attendance> findByEmployeeIdAndDateBetween(String employeeId, LocalDate startDate, LocalDate endDate);
//
//    List<Attendance> findByDate(LocalDate date);
//
//    List<Attendance> findByDateBetween(LocalDate startDate, LocalDate endDate);
//
//    // Status-based queries
//    List<Attendance> findByEmployeeIdAndStatus(String employeeId, String status);
//
//    List<Attendance> findByStatus(String status);
//
//    List<Attendance> findByDateAndStatus(LocalDate date, String status);
//
//    List<Attendance> findByDateBetweenAndStatus(LocalDate startDate, LocalDate endDate, String status);
//
//    // Combined queries for employee, date range, and status
//    List<Attendance> findByEmployeeIdAndDateBetweenAndStatus(String employeeId, LocalDate startDate, LocalDate endDate, String status);
//
//    // Time-based queries
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.checkinTime IS NOT NULL")
//    List<Attendance> findByEmployeeIdWithCheckinTime(@Param("employeeId") String employeeId);
//
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.checkoutTime IS NOT NULL")
//    List<Attendance> findByEmployeeIdWithCheckoutTime(@Param("employeeId") String employeeId);
//
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.checkinTime IS NOT NULL AND a.checkoutTime IS NOT NULL")
//    List<Attendance> findByEmployeeIdWithBothTimes(@Param("employeeId") String employeeId);
//
//    // Late arrival queries
//    @Query("SELECT a FROM Attendance a WHERE a.status = 'late' AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findLateArrivalsBetweenDates(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.status = 'late' AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findEmployeeLateArrivalsBetweenDates(@Param("employeeId") String employeeId, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    // Permission-based queries
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.permissionFrom IS NOT NULL AND a.permissionTo IS NOT NULL")
//    List<Attendance> findByEmployeeIdWithPermissionTimes(@Param("employeeId") String employeeId);
//
//    @Query("SELECT a FROM Attendance a WHERE a.status = 'permission' AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findPermissionRecordsBetweenDates(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    // Half-day queries
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.halfdayFrom IS NOT NULL AND a.halfdayTo IS NOT NULL")
//    List<Attendance> findByEmployeeIdWithHalfdayTimes(@Param("employeeId") String employeeId);
//
//    @Query("SELECT a FROM Attendance a WHERE a.status = 'halfday' AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findHalfdayRecordsBetweenDates(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.halfdayType = :halfdayType")
//    List<Attendance> findByEmployeeIdAndHalfdayType(@Param("employeeId") String employeeId, @Param("halfdayType") String halfdayType);
//
//    // Statistical queries
//    @Query("SELECT COUNT(a) FROM Attendance a WHERE a.employeeId = :employeeId AND a.status = :status")
//    Long countByEmployeeIdAndStatus(@Param("employeeId") String employeeId, @Param("status") String status);
//
//    @Query("SELECT COUNT(a) FROM Attendance a WHERE a.employeeId = :employeeId AND a.date BETWEEN :startDate AND :endDate")
//    Long countByEmployeeIdAndDateBetween(@Param("employeeId") String employeeId, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    @Query("SELECT COUNT(a) FROM Attendance a WHERE a.employeeId = :employeeId AND a.status = :status AND a.date BETWEEN :startDate AND :endDate")
//    Long countByEmployeeIdAndStatusAndDateBetween(@Param("employeeId") String employeeId, @Param("status") String status, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    // Monthly and yearly queries
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND YEAR(a.date) = :year AND MONTH(a.date) = :month")
//    List<Attendance> findByEmployeeIdAndYearAndMonth(@Param("employeeId") String employeeId, @Param("year") int year, @Param("month") int month);
//
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND YEAR(a.date) = :year")
//    List<Attendance> findByEmployeeIdAndYear(@Param("employeeId") String employeeId, @Param("year") int year);
//
//    // Department-based queries
//    List<Attendance> findByDepartment(String department);
//
//    List<Attendance> findByDepartmentAndDate(String department, LocalDate date);
//
//    List<Attendance> findByDepartmentAndDateBetween(String department, LocalDate startDate, LocalDate endDate);
//
//    List<Attendance> findByDepartmentAndStatus(String department, String status);
//
//    // Designation-based queries
//    List<Attendance> findByDesignation(String designation);
//
//    List<Attendance> findByDesignationAndDate(String designation, LocalDate date);
//
//    List<Attendance> findByDesignationAndDateBetween(String designation, LocalDate startDate, LocalDate endDate);
//
//    // Complex queries for analysis
//    @Query("SELECT a FROM Attendance a WHERE a.checkinTime > :lateTime AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findLateCheckinsAfterTime(@Param("lateTime") LocalTime lateTime, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    @Query("SELECT a FROM Attendance a WHERE a.checkoutTime < :earlyTime AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findEarlyCheckoutsBeforeTime(@Param("earlyTime") LocalTime earlyTime, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.checkinTime IS NULL AND a.status != 'absent' AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findMissingCheckinRecords(@Param("employeeId") String employeeId, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.checkoutTime IS NULL AND a.checkinTime IS NOT NULL AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findMissingCheckoutRecords(@Param("employeeId") String employeeId, @Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    // Today's specific queries
//    @Query("SELECT a FROM Attendance a WHERE a.date = CURRENT_DATE")
//    List<Attendance> findTodayAttendance();
//
//    @Query("SELECT a FROM Attendance a WHERE a.date = CURRENT_DATE AND a.status = :status")
//    List<Attendance> findTodayAttendanceByStatus(@Param("status") String status);
//
//    @Query("SELECT COUNT(a) FROM Attendance a WHERE a.date = CURRENT_DATE AND a.status = :status")
//    Long countTodayAttendanceByStatus(@Param("status") String status);
//
//    // Reason-based queries
//    List<Attendance> findByReason(String reason);
//
//    List<Attendance> findByEmployeeIdAndReason(String employeeId, String reason);
//
//    @Query("SELECT a FROM Attendance a WHERE a.reason IS NOT NULL AND a.reason != '' AND a.date BETWEEN :startDate AND :endDate")
//    List<Attendance> findRecordsWithReasonBetweenDates(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    // Custom queries for specific business logic
//    @Query("SELECT DISTINCT a.employeeId FROM Attendance a WHERE a.status IN ('present', 'late', 'onduty') AND a.date BETWEEN :startDate AND :endDate GROUP BY a.employeeId HAVING COUNT(a) = :requiredDays")
//    List<String> findEmployeesWithPerfectAttendance(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate, @Param("requiredDays") long requiredDays);
//
//    @Query("SELECT a.employeeId, COUNT(a) as absentCount FROM Attendance a WHERE a.status = 'absent' AND a.date BETWEEN :startDate AND :endDate GROUP BY a.employeeId HAVING COUNT(a) > :threshold")
//    List<Object[]> findEmployeesWithHighAbsenteeism(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate, @Param("threshold") long threshold);
//}