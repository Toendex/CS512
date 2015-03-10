//
//  myPoint.h
//  CS512-assignment2
//
//  Created by T on 3/2/14.
//  Copyright (c) 2014 T. All rights reserved.
//

#ifndef __CS512_assignment2__myPoint__
#define __CS512_assignment2__myPoint__

#include <iostream>

class myPoint {
public:
    int row;
    int col;
    double value;
    myPoint();
    myPoint(int,int,double);
    myPoint(const myPoint &);
    myPoint(myPoint *);
};
bool cmpMyPoint(const myPoint &a, const myPoint &b);

#endif /* defined(__CS512_assignment2__myPoint__) */
