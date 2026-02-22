public class leetcode1(twosum) {
    public int[] twosum(int[] nums, int target){
        for (int i=0;i<nums.length;i++){
            for(int j=0;j<nums.length;j++){
                if(nums[i]+nums[j]==target){
                    return new int[]{i,j};
                }
            }
        }
        return null;
    }
    
}
public static void main(string[] args){
    int nums[]={2,7,5,11};
    int target=9;
    int result[] = solution.twosum(nums,target);


         if (result != null) {
            System.out.println("Indices: " + Arrays.toString(result));
            System.out.println("Values: " + nums[result[0]] + " + " + nums[result[1]]);
        } else {
            System.out.println("No solution found");
        }
    }
}