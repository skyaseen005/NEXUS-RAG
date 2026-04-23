    int findPeakElement(vector<int>& arr) { 
         
        int n = arr.size(); 
        if(n == 1){ 
            return 0; 
        } 
        int low = 0, high = n - 1; 
        while (low <= high){ 
             
            int mid = low + (high - low)/2; 
            if(mid == 0){ 
                if(arr[mid] > arr[mid + 1]){ 
                    return mid; 
                }else{ 
                    low = mid + 1; 
                } 
             
            }else if(mid == n - 1){ 
                if(arr[mid - 1] < arr[mid]){ 
                    return mid;     
                }else{ 
                    high = mid - 1; 
                } 
                  
            }else{ 
                if(arr[mid] > arr[mid - 1] && arr[mid] > arr[mid + 1]){ 
                    return mid; 
                } 
                else if(arr[mid] < arr[mid + 1]){ 
                    low = mid + 1; 
                }else{ 
                    high = mid - 1; 
                } 
            } 
        }

            // Handle small numbers directly
        if (x < 2) return x;

        // Initialize binary search range
        int left = 1, right = x / 2, ans = 0;

        // Perform binary search
        while (left <= right) {
            // Find middle point
            long long mid = left + (right - left) / 2;

            // Check if mid*mid is less than or equal to x
            if (mid * mid <= x) {
                // Store mid as potential answer
                ans = mid;
                // Move to right half
                left = mid + 1;
            } else {
                // Move to left half
                right = mid - 1;
            }
        }

        // Return final answer
        return ans;
    }