//
//  myPoint.cpp
//  CS512-assignment2
//
//  Created by T on 3/2/14.
//  Copyright (c) 2014 T. All rights reserved.
//

#include "myPoint.h"

myPoint::myPoint() {
    this->row=0;
    this->col=0;
    this->value=0;
}

myPoint::myPoint(int r,int c,double v) {
    this->row=r;
    this->col=c;
    this->value=v;
}

myPoint::myPoint(const myPoint &a) {
    this->row=a.row;
    this->col=a.col;
    this->value=a.value;

}

myPoint::myPoint(myPoint *a) {
    this->row=a->row;
    this->col=a->col;
    this->value=a->value;
}

bool cmpMyPoint(const myPoint &a, const myPoint &b) {
    return a.value > b.value;
}