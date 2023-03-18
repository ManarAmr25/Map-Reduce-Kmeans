package org.example;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeans {

    final static float threshold = 0.0001f;
    final static String nKey = "features.n";
    final static String kKey = "k";

    public static class KmeansMapper
            extends Mapper<LongWritable, Text, IntWritable, PointWritable>{

        private int k;
        private int n;
        private float[][] centroids;

        @Override
        protected void setup(Context context) {
            this.k = context.getConfiguration().getInt(KMeans.kKey, 0);
            this.n = context.getConfiguration().getInt(KMeans.nKey, 0);

            this.centroids = KMeans.getCentroidsFromConf(context.getConfiguration());
            System.out.println("end setup mapper ");
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString(), ",");
            float[] point = new float[this.n];
            for (int i = 0; i < this.n; i++) {
                float feature = Float.parseFloat(itr.nextToken());
                point[i] = feature;
            }
            PointWritable p = new PointWritable(point);
            IntWritable index = assignClass(point);
            context.write(index, p);
        }

        private IntWritable assignClass(float[] point) {
            double min = Double.MAX_VALUE;
            int classId = 0;
            for (int i = 0; i < this.k; i++) {
                double dist = 0;
                for (int j = 0; j < this.n; j++)
                    dist += Math.pow((centroids[i][j] - point[j]), 2);

                dist = Math.sqrt(dist);
                if (dist < min) {
                    min = dist;
                    classId = i;
                }
            }
            return new IntWritable(classId);
        }
    }

    public static class KmeansCombiner
            extends Reducer<IntWritable, PointWritable, IntWritable, PointWritable> {

        private int n;
        @Override
        protected void setup(Context context) {

            this.n = context.getConfiguration().getInt(KMeans.nKey, 0);
            System.out.println("end setup combiner");
        }
        @Override
        protected void reduce(IntWritable key, Iterable<PointWritable> values, Context context)
                throws IOException, InterruptedException {
            float[] sum = new float[this.n];
            int count = 0;

            for (PointWritable point : values) {
                count++;
                for (int i = 0; i < this.n; i++) {
                    sum[i] += point.getFeatures()[i];
                }
            }

            PointWritable result = new PointWritable(sum, count);

            context.write(key, result);
        }
    }

    public static class KmeansReducer
            extends Reducer<IntWritable, PointWritable, IntWritable, PointWritable> {

        private int n;
        @Override
        protected void setup(Context context) {
            this.n = context.getConfiguration().getInt(KMeans.nKey, 0);
            System.out.println("end setup reducer");
        }

        @Override
        protected void reduce(IntWritable key, Iterable<PointWritable> values, Context context) throws IOException, InterruptedException {

            float[] centroid = new float[this.n];
            int count = 0;

            for (PointWritable point : values) {
                count += point.getCount();

                for (int i = 0; i < this.n; i++) {
                    centroid[i] += point.getFeatures()[i];
                }
            }
            for (int i = 0; i < this.n; i++) {
                centroid[i] /= count;
            }

            PointWritable result = new PointWritable(centroid);

            context.write(key, result);
        }
    }

    protected static void setCentroidsFromFile(Configuration conf, String path) throws IOException {
        int n = conf.getInt(KMeans.nKey, 4);
        Path pt = new Path(path);
        FileSystem fs = FileSystem.get(conf);
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
        try {
            String line;
            line=br.readLine();
            while (line != null){
                String[] arr = line.split("(\\s+)|(\\s*,+\\s*)");
                String key = "c" + arr[0];
                conf.setStrings(key, Arrays.copyOfRange(arr, 1, n + 1));
                line = br.readLine();
            }
        } finally {
            br.close();
        }
    }

    protected static float[][] getCentroidsFromConf(Configuration conf){
        int k = conf.getInt(KMeans.kKey, 0);
        int n = conf.getInt(KMeans.nKey, 0);
        float[][] centroids = new float[k][n];

        for (int i = 0; i < k; i++) {
            String[] centroid = conf.getStrings("c" + i);

            for (int j = 0; j < n; j++)
                centroids[i][j] = Float.parseFloat(centroid[j]);
        }
        return centroids;
    }
    
    protected static boolean isConverged(Configuration conf, float[][] oldCentroids, float[][] newCentroids){
        int k = conf.getInt(KMeans.kKey, 0);
        int n = conf.getInt(KMeans.nKey, 0);

        for (int i = 0; i < k; i++) {
            double distance = 0;
            for (int j = 0; j < n; j++) {
                distance += Math.pow((oldCentroids[i][j] - newCentroids[i][j]), 2);
            }
            distance = Math.sqrt(distance);
            if (distance > threshold) return false;
        }
        return true;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Start K-means ...");
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://localhost:9000");
        conf.setInt("mapreduce.task.io.sort.mb", 200);
        conf.setInt(KMeans.nKey, Integer.parseInt(args[0]));
        conf.setInt(KMeans.kKey, Integer.parseInt(args[1]));


        KMeans.setCentroidsFromFile(conf, args[2]);

        Path input = new Path(args[3]);
        Path output = new Path(args[4]);

        int countSteps = 0;
        long startTime = System.nanoTime();
        float[][] oldCentroids = getCentroidsFromConf(conf);
        boolean isConverged = false;
        while (!isConverged){
            System.out.println("* ** *** iteration " + countSteps++ + " *** ** *");
            if (output.getFileSystem(conf).exists(output))
                output.getFileSystem(conf).delete(output);
            Job job = Job.getInstance(conf, "Kmeans_" + (countSteps-1));
            job.setJarByClass(KMeans.class);
            job.setMapperClass(KmeansMapper.class);
            job.setCombinerClass(KmeansCombiner.class);
            job.setReducerClass(KmeansReducer.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(PointWritable.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(PointWritable.class);
            job.setNumReduceTasks(1);

            FileInputFormat.addInputPath(job, input);
            FileOutputFormat.setOutputPath(job, output);
            job.waitForCompletion(true);

            //update centroids
            KMeans.setCentroidsFromFile(conf, output + "/part-r-00000");
            float[][] newCentroids = KMeans.getCentroidsFromConf(conf);
            isConverged = KMeans.isConverged(conf, oldCentroids, newCentroids);
            oldCentroids = newCentroids;
        }

        long endTime   = System.nanoTime();
        long totalTime = endTime - startTime;
        System.out.println("Total Runtime: " + (totalTime/1e9) + " sec");
        System.out.println("Count Steps: " + countSteps);

    }
}