//package com.Attendance.BackEnd_Attendance.Controller;
//
//import com.Attendance.BackEnd_Attendance.Model.Attendance;
//import com.Attendance.BackEnd_Attendance.Service.AttendanceConditionService;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.format.annotation.DateTimeFormat;
//import org.springframework.http.ResponseEntity;
//import org.springframework.web.bind.annotation.*;
//
//import java.time.LocalDate;
//import java.util.List;
//import java.util.Map;
//
//@RestController
//@RequestMapping("/api/attendance-conditions")
//@CrossOrigin(origins = "*")
//public class AttendanceConditionController {
//
//    @Autowired
//    private AttendanceConditionService attendanceConditionService;
//
//    /**
//     * Get attendance statistics for an employee
//     */
//    @GetMapping("/{employeeId}/stats")
//    public ResponseEntity<com.Attendance.BackEnd_Attendance.Model.AttendanceStatsDTO> getAttendanceStats(@PathVariable String employeeId) {
//        com.Attendance.BackEnd_Attendance.Model.AttendanceStatsDTO stats = attendanceConditionService.getAttendanceStatistics(employeeId);
//        return ResponseEntity.ok(stats);
//    }
//
//    /**
//     * Get today's attendance status for an employee
//     */
//    @GetMapping("/{employeeId}/today")
//    public ResponseEntity<Attendance> getTodayAttendance(@PathVariable String employeeId) {
//        Attendance todayAttendance = attendanceConditionService.getTodayAttendance(employeeId);
//        if (todayAttendance != null) {
//            return ResponseEntity.ok(todayAttendance);
//        }
//        return ResponseEntity.notFound().build();
//    }
//
//    /**
//     * Get filtered attendance records for an employee
//     */
//    @GetMapping("/{employeeId}")
//    public ResponseEntity<List<Attendance>> getFilteredAttendance(
//            @PathVariable String employeeId,
//            @RequestParam(required = false) String filter,
//            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
//            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {
//
//        List<Attendance> attendanceRecords = attendanceConditionService.getFilteredAttendance(
//                employeeId, filter, startDate, endDate);
//        return ResponseEntity.ok(attendanceRecords);
//    }
//
//    /**
//     * Get attendance records by status
//     */
//    @GetMapping("/{employeeId}/status/{status}")
//    public ResponseEntity<List<Attendance>> getAttendanceByStatus(
//            @PathVariable String employeeId,
//            @PathVariable String status) {
//
//        List<Attendance> attendanceRecords = attendanceConditionService.getAttendanceByStatus(employeeId, status);
//        return ResponseEntity.ok(attendanceRecords);
//    }
//
//    /**
//     * Get attendance records for date range
//     */
//    @GetMapping("/{employeeId}/daterange")
//    public ResponseEntity<List<Attendance>> getAttendanceByDateRange(
//            @PathVariable String employeeId,
//            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
//            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {
//
//        List<Attendance> attendanceRecords = attendanceConditionService.getAttendanceByDateRange(
//                employeeId, startDate, endDate);
//        return ResponseEntity.ok(attendanceRecords);
//    }
//
//    /**
//     * Get monthly attendance summary
//     */
//    @GetMapping("/{employeeId}/monthly-summary")
//    public ResponseEntity<Map<String, Integer>> getMonthlyAttendanceSummary(
//            @PathVariable String employeeId,
//            @RequestParam int year,
//            @RequestParam int month) {
//
//        Map<String, Integer> summary = attendanceConditionService.getMonthlyAttendanceSummary(employeeId, year, month);
//        return ResponseEntity.ok(summary);
//    }
//
//    /**
//     * Get weekly attendance summary
//     */
//    @GetMapping("/{employeeId}/weekly-summary")
//    public ResponseEntity<Map<String, Integer>> getWeeklyAttendanceSummary(
//            @PathVariable String employeeId,
//            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate weekStartDate) {
//
//        Map<String, Integer> summary = attendanceConditionService.getWeeklyAttendanceSummary(employeeId, weekStartDate);
//        return ResponseEntity.ok(summary);
//    }
//
//    /**
//     * Check if employee is present today
//     */
//    @GetMapping("/{employeeId}/is-present-today")
//    public ResponseEntity<Boolean> isEmployeePresentToday(@PathVariable String employeeId) {
//        boolean isPresent = attendanceConditionService.isEmployeePresentToday(employeeId);
//        return ResponseEntity.ok(isPresent);
//    }
//
//    /**
//     * Get employees with specific attendance status for today
//     */
//    @GetMapping("/today/status/{status}")
//    public ResponseEntity<List<Attendance>> getEmployeesByTodayStatus(@PathVariable String status) {
//        List<Attendance> attendanceRecords = attendanceConditionService.getEmployeesByTodayStatus(status);
//        return ResponseEntity.ok(attendanceRecords);
//    }
//
//    /**
//     * Get attendance statistics for all employees
//     */
//    @GetMapping("/all-employees/stats")
//    public ResponseEntity<Map<String, com.Attendance.BackEnd_Attendance.Model.AttendanceStatsDTO>> getAllEmployeesStats() {
//        Map<String, com.Attendance.BackEnd_Attendance.Model.AttendanceStatsDTO> allStats = attendanceConditionService.getAllEmployeesAttendanceStats();
//        return ResponseEntity.ok(allStats);
//    }
//
//    /**
//     * Get late arrivals for a specific date range
//     */
//    @GetMapping("/late-arrivals")
//    public ResponseEntity<List<Attendance>> getLateArrivals(
//            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
//            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {
//
//        List<Attendance> lateArrivals = attendanceConditionService.getLateArrivals(startDate, endDate);
//        return ResponseEntity.ok(lateArrivals);
//    }
//
//    /**
//     * Get employees with perfect attendance for a month
//     */
//    @GetMapping("/perfect-attendance")
//    public ResponseEntity<List<String>> getEmployeesWithPerfectAttendance(
//            @RequestParam int year,
//            @RequestParam int month) {
//
//        List<String> employeeIds = attendanceConditionService.getEmployeesWithPerfectAttendance(year, month);
//        return ResponseEntity.ok(employeeIds);
//    }
//}
