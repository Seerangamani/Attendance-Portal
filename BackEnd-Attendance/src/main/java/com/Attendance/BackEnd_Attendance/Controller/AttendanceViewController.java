package com.Attendance.BackEnd_Attendance.Controller;

import com.Attendance.BackEnd_Attendance.Model.Attendance;
import com.Attendance.BackEnd_Attendance.Service.AttendanceViewService;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/view-attendance")
@CrossOrigin(origins = "*")
public class AttendanceViewController {

    private final AttendanceViewService attendanceViewService;

    public AttendanceViewController(AttendanceViewService attendanceViewService) {
        this.attendanceViewService = attendanceViewService;
    }

    // 1. Get all employees with today's status
    @GetMapping("/today")
    public List<Map<String, Object>> getAllEmployeesWithTodayStatus() {
        return attendanceViewService.getEmployeeAttendanceForDate(LocalDate.now());
    }

    // 2. Get today's status for a specific employee
    @GetMapping("/today/employee/{id}")
    public Map<String, Object> getTodayStatusForEmployee(@PathVariable String id) {
        return attendanceViewService.getTodayStatusForEmployee(id);
    }

    // 3. Get full attendance for a specific employee
    @GetMapping("/employee/{id}")
    public List<Attendance> getAttendanceByEmployeeId(@PathVariable String id) {
        return attendanceViewService.getFullAttendanceForEmployee(id);
    }

    // 4. Filter attendance by date range
    @GetMapping("/filter")
    public List<Attendance> filterAttendance(
            @RequestParam String id,
            @RequestParam(required = false) String fromDate,
            @RequestParam(required = false) String toDate
    ) {
        return attendanceViewService.filterAttendance(id, fromDate, toDate);
    }

    // 5. Filter attendance with status counts and percentage
    @GetMapping("/filter/with-status-counts")
    public Map<String, Object> filterAttendanceWithCounts(
            @RequestParam String id,
            @RequestParam(required = false) String fromDate,
            @RequestParam(required = false) String toDate
    ) {
        List<Attendance> filteredList = attendanceViewService.filterAttendance(id, fromDate, toDate);

        // Count each status
        Map<String, Long> rawCounts = filteredList.stream()
                .filter(a -> a.getStatus() != null && !a.getStatus().trim().isEmpty())
                .collect(Collectors.groupingBy(
                        a -> a.getStatus().toLowerCase().trim(),
                        Collectors.counting()
                ));

        // Ensure all expected statuses are present with 0 count if not found
        String[] expectedStatuses = {"present", "absent", "late", "halfday", "permission", "onduty"};
        Map<String, Long> statusCounts = new HashMap<>();
        for (String status : expectedStatuses) {
            statusCounts.put(status, rawCounts.getOrDefault(status, 0L));
        }

        // Calculate percentage
        long totalRecords = filteredList.size();
        long presentCount = statusCounts.get("present") + statusCounts.get("late");
        double percentage = totalRecords > 0 ? Math.round((presentCount * 100.0) / totalRecords * 100.0) / 100.0 : 0.0;

        Map<String, Object> result = new HashMap<>();
        result.put("attendance", filteredList);
        result.put("statusCounts", statusCounts);
        result.put("totalRecords", totalRecords);
        result.put("percentage", percentage);
        return result;
    }

    // 6. Get overall percentage for an employee
    @GetMapping("/overall-percentage")
    public Map<String, Object> getOverallPresentPercentage(@RequestParam String id) {
        List<Attendance> allAttendance = attendanceViewService.getFullAttendanceForEmployee(id);
        long totalRecords = allAttendance.size();

        // Count present and late as attendance (both are considered as attended)
        long presentCount = allAttendance.stream()
                .filter(a -> a.getStatus() != null)
                .filter(a -> {
                    String status = a.getStatus().toLowerCase().trim();
                    return status.equals("present") || status.equals("late");
                })
                .count();

        double percentage = totalRecords > 0 ? Math.round((presentCount * 100.0) / totalRecords * 100.0) / 100.0 : 0.0;

        Map<String, Object> response = new HashMap<>();
        response.put("percentage", percentage);
        response.put("totalRecords", totalRecords);
        response.put("presentCount", presentCount);
        return response;
    }

    // 7. Get monthly attendance summary for dropdown
    @GetMapping("/months/{id}")
    public List<Map<String, Object>> getAvailableMonths(@PathVariable String id) {
        return attendanceViewService.getAvailableMonthsForEmployee(id);
    }

    // 8. Get attendance statistics for dashboard
    @GetMapping("/statistics/{id}")
    public Map<String, Object> getAttendanceStatistics(@PathVariable String id) {
        return attendanceViewService.getAttendanceStatistics(id);
    }
}