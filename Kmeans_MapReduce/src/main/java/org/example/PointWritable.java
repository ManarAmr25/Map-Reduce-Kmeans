package org.example;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class PointWritable implements Writable {
//    private FloatWritable f1;
//    private FloatWritable f2;
//    private FloatWritable f3;
//    private FloatWritable f4;
    private int count;
    private int n;
    private float[] features;

//    public PointWritable() {
//        this.f1 = new FloatWritable(0);
//        this.f2 = new FloatWritable(0);
//        this.f3 = new FloatWritable(0);
//        this.f4 = new FloatWritable(0);
//        this.count = new IntWritable(0);
//    }

    public PointWritable(){
        this.features = null;
        this.n = 0;
        this.count = 0;
    }
    public PointWritable(float[] features) {
        this.features = features;
        this.count = 0;
        this.n = features.length;
    }
    public PointWritable(float[] features, int count) {
        this.features = features;
        this.count = count;
        this.n = features.length;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
//        f1.write(out);
//        f2.write(out);
//        f3.write(out);
//        f4.write(out);
//        count.write(out);

        dataOutput.writeInt(this.n);
        dataOutput.writeInt(this.count);

        for (int i = 0; i < this.n; i++) {
            dataOutput.writeFloat(this.features[i]);
        }
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
//        f1.readFields(dataInput);
//        f2.readFields(dataInput);
//        f3.readFields(dataInput);
//        f4.readFields(dataInput);
//        count.readFields(dataInput);

        this.n = dataInput.readInt();
        this.count = dataInput.readInt();
        this.features = new float[this.n];

        for (int i = 0; i < this.n; i++) {
            this.features[i] = dataInput.readFloat();
        }
    }

    @Override
    public String toString() {
        String res = "";
        for (int i = 0; i < this.features.length; i++) {
            res += this.features[i] + "\t";
        }
        res += this.count;
        return res;
    }

    public int getCount() {
        return count;
    }

    public float[] getFeatures() {
        return features;
    }
}


